//! Headless backend for virtual outputs and testing.
//!
//! This backend can be used for:
//! - Running niri without physical displays (for VNC, screensharing, etc.)
//! - Testing purposes
//!
//! Note: This is missing some parts like direct rendering to dmabufs.

use std::mem;
use std::os::fd::OwnedFd;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::Context as _;
use niri_config::OutputName;
use smithay::backend::allocator::dmabuf::Dmabuf;
use smithay::backend::allocator::gbm::GbmDevice;
use smithay::backend::drm::{DrmDeviceFd, DrmNode};
use smithay::backend::libinput::{LibinputInputBackend, LibinputSessionInterface};
use smithay::backend::session::{AsErrno, Session};
use smithay::reexports::calloop::timer::{TimeoutAction, Timer};
use smithay::backend::egl::{EGLContext, EGLDisplay};
use smithay::backend::renderer::element::RenderElementStates;
use smithay::backend::renderer::gles::GlesRenderer;
use smithay::backend::renderer::ImportDma;
use smithay::reexports::input::Libinput;
use smithay::output::{Mode, Output, PhysicalProperties, Subpixel};
use smithay::reexports::rustix::fs::{open, Mode as RustixMode, OFlags};
use smithay::reexports::rustix::io::Errno as RustixErrno;
use smithay::reexports::wayland_protocols::wp::presentation_time::server::wp_presentation_feedback;
use smithay::utils::{DeviceFd, Size};
use smithay::wayland::dmabuf::{DmabufFeedbackBuilder, DmabufGlobal};
use smithay::wayland::presentation::Refresh;

use super::{IpcOutputMap, OutputId, RenderResult};
use crate::niri::{Niri, RedrawState, State};
use crate::render_helpers::{resources, shaders};
use crate::utils::{get_monotonic_time, logical_output};

pub struct Headless {
    renderer: Option<GlesRenderer>,
    render_node: Option<DrmNode>,
    dmabuf_global: Option<DmabufGlobal>,
    _libinput: Option<Libinput>,
    ipc_outputs: Arc<Mutex<IpcOutputMap>>,
    virtual_outputs: super::VirtualOutputs,
}

/// A minimal session that opens input devices directly. Used by the headless
/// backend because it does not run on a VT session (where libseat would work),
/// and we still need to read from uinput devices created by clients such as
/// Sunshine. Requires the user to be in the `input` group.
struct DirectSession;

#[derive(Debug)]
struct DirectSessionError(RustixErrno);

impl AsErrno for DirectSessionError {
    fn as_errno(&self) -> Option<i32> {
        Some(self.0.raw_os_error())
    }
}

impl Session for DirectSession {
    type Error = DirectSessionError;

    fn open(&mut self, path: &Path, flags: OFlags) -> Result<OwnedFd, Self::Error> {
        open(path, flags, RustixMode::empty()).map_err(DirectSessionError)
    }

    fn close(&mut self, fd: OwnedFd) -> Result<(), Self::Error> {
        drop(fd);
        Ok(())
    }

    fn change_vt(&mut self, _vt: i32) -> Result<(), Self::Error> {
        Ok(())
    }

    fn is_active(&self) -> bool {
        true
    }

    fn seat(&self) -> String {
        "seat0".to_owned()
    }
}

impl Headless {
    pub fn new() -> Self {
        Self {
            renderer: None,
            render_node: None,
            dmabuf_global: None,
            _libinput: None,
            ipc_outputs: Default::default(),
            virtual_outputs: super::VirtualOutputs::new(),
        }
    }

    pub fn init(&mut self, niri: &mut Niri) {
        // Listen for input devices via libinput so that virtual input devices
        // (e.g. Sunshine's uinput keyboards/mice) reach this compositor.
        let mut libinput =
            Libinput::new_with_udev(LibinputSessionInterface::from(DirectSession));
        if let Err(()) = libinput.udev_assign_seat("seat0") {
            warn!("error assigning seat0 to libinput");
            return;
        }
        let input_backend = LibinputInputBackend::new(libinput.clone());
        niri
            .event_loop
            .insert_source(input_backend, |mut event, _, state| {
                state.process_libinput_event(&mut event);
                state.process_input_event(event);
            })
            .unwrap();
        self._libinput = Some(libinput);
    }

    pub fn add_renderer(&mut self, niri: &mut Niri) -> anyhow::Result<()> {
        if self.renderer.is_some() {
            error!("add_renderer: renderer must not already exist");
            return Ok(());
        }

        // Bind the discrete GPU (renderD128) directly so that rendered buffers are
        // allocated by the dGPU. This lets clients (e.g. Sunshine) use zero-copy
        // dmabuf import + VAAPI encode on the same device.
        let render_node = DrmNode::from_path("/dev/dri/renderD128")
            .context("error getting dGPU render node (is /dev/dri/renderD128 present?)")?;
        let render_path = render_node
            .dev_path()
            .ok_or_else(|| anyhow::anyhow!("dGPU render node has no dev path"))?;
        let fd = open(
            &render_path,
            OFlags::RDWR | OFlags::CLOEXEC | OFlags::NONBLOCK,
            RustixMode::empty(),
        )
        .context("error opening dGPU render node")?;
        let device_fd = DrmDeviceFd::new(DeviceFd::from(fd));
        let gbm = GbmDevice::new(device_fd).context("error creating GBM device")?;

        let mut renderer = unsafe {
            let display = EGLDisplay::new(gbm.clone()).context("error creating EGL display")?;
            let context = EGLContext::new(&display).context("error creating EGL context")?;
            GlesRenderer::new(context).context("error creating renderer")?
        };

        resources::init(&mut renderer);
        shaders::init(&mut renderer);

        // Create the linux-dmabuf global with the dGPU as the primary device. This is
        // what lets a client resolve the render node to create its EGL display.
        let formats = renderer.dmabuf_formats();
        let default_feedback = DmabufFeedbackBuilder::new(render_node.dev_id(), formats)
            .build()
            .context("error building default dmabuf feedback")?;
        let dmabuf_global = niri
            .dmabuf_state
            .create_global_with_default_feedback::<State>(&niri.display_handle, &default_feedback);

        self.render_node = Some(render_node);
        self.dmabuf_global = Some(dmabuf_global);
        self.renderer = Some(renderer);
        Ok(())
    }

    /// Add an output for testing (uses lowercase naming like `headless-1`).
    /// This is kept for backwards compatibility with tests.
    pub fn add_output(&mut self, niri: &mut Niri, n: u8, size: (u16, u16)) {
        let connector = format!("headless-{n}");
        let make = "niri".to_string();
        let model = "headless".to_string();
        let serial = n.to_string();

        let output = Output::new(
            connector.clone(),
            PhysicalProperties {
                size: (0, 0).into(),
                subpixel: Subpixel::Unknown,
                make: make.clone(),
                model: model.clone(),
                serial_number: serial.clone(),
            },
        );

        let mode = Mode {
            size: Size::from((i32::from(size.0), i32::from(size.1))),
            refresh: 60_000,
        };
        output.change_current_state(Some(mode), None, None, None);
        output.set_preferred(mode);

        output.user_data().insert_if_missing(|| OutputName {
            connector,
            make: Some(make),
            model: Some(model),
            serial: Some(serial),
        });

        let physical_properties = output.physical_properties();
        self.ipc_outputs.lock().unwrap().insert(
            OutputId::next(),
            niri_ipc::Output {
                name: output.name(),
                make: physical_properties.make,
                model: physical_properties.model,
                serial: None,
                physical_size: None,
                modes: vec![niri_ipc::Mode {
                    width: size.0,
                    height: size.1,
                    refresh_rate: 60_000,
                    is_preferred: true,
                }],
                current_mode: Some(0),
                is_custom_mode: true,
                vrr_supported: false,
                vrr_enabled: false,
                logical: Some(logical_output(&output)),
            },
        );

        niri.add_output(output, None, false);
    }

    /// Create a virtual headless output with the given dimensions.
    /// Returns the name of the created output (e.g., "HEADLESS-1").
    pub fn create_virtual_output(
        &mut self,
        niri: &mut Niri,
        width: u16,
        height: u16,
        refresh_rate: u32,
    ) -> Result<String, String> {
        self.virtual_outputs
            .create(niri, &self.ipc_outputs, width, height, refresh_rate)
    }

    pub fn remove_virtual_output(&mut self, niri: &mut Niri, name: &str) -> Result<(), String> {
        self.virtual_outputs
            .remove(niri, &self.ipc_outputs, name)
    }

    pub fn seat_name(&self) -> String {
        "headless".to_owned()
    }

    pub fn with_primary_renderer<T>(
        &mut self,
        f: impl FnOnce(&mut GlesRenderer) -> T,
    ) -> Option<T> {
        self.renderer.as_mut().map(f)
    }

    pub fn render(&mut self, niri: &mut Niri, output: &Output) -> RenderResult {
        let now = get_monotonic_time();

        let states = RenderElementStates::default();
        let mut presentation_feedbacks = niri.take_presentation_feedbacks(output, &states);
        presentation_feedbacks.presented::<_, smithay::utils::Monotonic>(
            now,
            Refresh::Unknown,
            0,
            wp_presentation_feedback::Kind::empty(),
        );

        let output_state = niri.output_state.get_mut(output).unwrap();
        match mem::replace(&mut output_state.redraw_state, RedrawState::Idle) {
            RedrawState::Idle => unreachable!(),
            RedrawState::Queued => (),
            RedrawState::WaitingForVBlank { .. } => unreachable!(),
            RedrawState::WaitingForEstimatedVBlank(token)
            | RedrawState::WaitingForEstimatedVBlankAndQueued(token) => {
                niri.event_loop.remove(token);
            }
        }

        // Update the frame clock so animation timing works correctly.
        output_state.frame_clock.presented(now);
        output_state.frame_callback_sequence = output_state.frame_callback_sequence.wrapping_add(1);

        // Use a timer to pace redraws, simulating vblank for headless outputs.
        let refresh_interval = output_state
            .frame_clock
            .refresh_interval()
            .unwrap_or(Duration::from_micros(16_667));

        let output_clone = output.clone();
        let timer = Timer::from_duration(refresh_interval);
        let token = niri
            .event_loop
            .insert_source(timer, move |_, _, data| {
                let output_state = data.niri.output_state.get_mut(&output_clone).unwrap();
                output_state.frame_callback_sequence =
                    output_state.frame_callback_sequence.wrapping_add(1);

                match mem::replace(&mut output_state.redraw_state, RedrawState::Idle) {
                    RedrawState::WaitingForEstimatedVBlank(_) => (),
                    RedrawState::WaitingForEstimatedVBlankAndQueued(_) => {
                        output_state.redraw_state = RedrawState::Queued;
                        return TimeoutAction::Drop;
                    }
                    _ => unreachable!(),
                }

                if output_state.unfinished_animations_remain {
                    data.niri.queue_redraw(&output_clone);
                } else {
                    data.niri.send_frame_callbacks_for_virtual_output(&output_clone);
                }
                TimeoutAction::Drop
            })
            .unwrap();
        output_state.redraw_state = RedrawState::WaitingForEstimatedVBlank(token);

        RenderResult::Submitted
    }

    pub fn import_dmabuf(&mut self, dmabuf: &Dmabuf) -> bool {
        let Some(renderer) = self.renderer.as_mut() else {
            return false;
        };
        match renderer.import_dmabuf(dmabuf, None) {
            Ok(_texture) => true,
            Err(err) => {
                debug!("error importing dmabuf: {err:?}");
                false
            }
        }
    }

    pub fn ipc_outputs(&self) -> Arc<Mutex<IpcOutputMap>> {
        self.ipc_outputs.clone()
    }
}

impl Default for Headless {
    fn default() -> Self {
        Self::new()
    }
}
