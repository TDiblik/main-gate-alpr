#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release

mod models;
mod utils;
mod websocket;

use eframe::{egui, emath::Align, epaint::Color32};
use std::{
    sync::{Arc, Mutex},
    thread,
    time::Duration,
};
use utils::calc_scaling_factor;

use models::{CarRows, SharedWebSocketState, WebSocketState, WebSocketStates};
use websocket::websocket_main;

const WEBSOCKET_URL: &str = "ws://localhost:8765";
const CAR_IMAGE_HEIGHT: f32 = 250.0;
const LICENSE_PLATE_IMAGE_HEIGHT: f32 = 75.0;
const TEXT_HEIGHT: f32 = 75.0;

fn main() -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions {
        initial_window_size: Some(egui::vec2(1000.0, 1000.0)),
        ..Default::default()
    };
    eframe::run_native(
        "Main gate ALPR client",
        options,
        Box::new(|_cc| Box::<App>::default()),
    )
}

struct App {
    car_rows: CarRows,
    websocket_state: SharedWebSocketState,
}

impl Default for App {
    fn default() -> Self {
        let new_app = Self {
            car_rows: Arc::new(Mutex::new(vec![])),
            websocket_state: Arc::new(Mutex::new(WebSocketState::default())),
        };

        let cloned_car_rows_state = Arc::clone(&new_app.car_rows);
        let cloned_websocket_state = Arc::clone(&new_app.websocket_state);
        // TODO: Read from env during compilation
        thread::spawn(move || {
            websocket_main(cloned_car_rows_state, cloned_websocket_state, WEBSOCKET_URL)
        });

        new_app
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        ctx.request_repaint_after(Duration::from_millis(500));
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.with_layout(egui::Layout::top_down(Align::RIGHT), |ui| {
                ui.horizontal(|ui| {
                    match &self.websocket_state.lock().unwrap().value {
                        WebSocketStates::Connected => {
                            ui.label(egui::RichText::from("Connected").color(Color32::GREEN))
                        }
                        WebSocketStates::Reconnecting => {
                            ui.label(egui::RichText::from("Reconnecting").color(Color32::YELLOW))
                        }
                        WebSocketStates::Closed(s) => ui
                            .label(egui::RichText::from("Connection closed").color(Color32::RED))
                            .on_hover_text(s),
                    };
                    ui.add_space(2.0);
                    ui.label("Websocket status: ");
                });
            });
            egui::ScrollArea::vertical().show(ui, |ui| {
                for car in self.car_rows.lock().unwrap().iter() {
                    ui.horizontal(|ui| {
                        ui.group(|ui| {
                            ui.vertical(|ui| {
                                car.car_image.show_scaled(
                                    ui,
                                    calc_scaling_factor(car.car_image.height(), CAR_IMAGE_HEIGHT),
                                );
                                ui.add_space(10.0);
                                car.license_plate_image.show_scaled(
                                    ui,
                                    calc_scaling_factor(
                                        car.license_plate_image.height(),
                                        LICENSE_PLATE_IMAGE_HEIGHT,
                                    ),
                                );
                            });

                            ui.centered_and_justified(|ui| {
                                ui.vertical_centered_justified(|ui| {
                                    ui.vertical_centered(|ui| {
                                        ui.label(
                                            egui::RichText::from(&car.license_plate_as_string)
                                                .size(TEXT_HEIGHT)
                                                .color(Color32::WHITE),
                                        );

                                        ui.label(
                                            egui::RichText::from(format!(
                                                "Received at: {}",
                                                car.received_at_formatted
                                            ))
                                            .color(Color32::WHITE),
                                        );
                                        ui.add_space(5.0);

                                        ui.label(format!("id: {}", car.uuid));
                                        ui.add_space(10.0);
                                    });
                                });
                            });
                        });
                    });

                    ui.add_space(32.0);
                }
            });
        });
    }
}
