#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release

mod models;
mod utils;
mod websocket;

use eframe::{egui, epaint::Color32};
use std::{
    sync::{Arc, Mutex},
    thread,
    time::Duration,
};
use utils::calc_scaling_factor;

use models::CarRows;
use websocket::websocket_main;

const CAR_IMAGE_HEIGHT: f32 = 300.0;
const LICENSE_PLATE_IMAGE_HEIGHT: f32 = 100.0;
const TEXT_HEIGHT: f32 = 80.0;

fn main() -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions {
        initial_window_size: Some(egui::vec2(1000.0, 1000.0)),
        ..Default::default()
    };
    eframe::run_native(
        "Main gate ALPR",
        options,
        Box::new(|_cc| Box::<App>::default()),
    )
}

struct App {
    car_rows: CarRows,
}

impl Default for App {
    fn default() -> Self {
        let new_app = Self {
            car_rows: Arc::new(Mutex::new(vec![])),
        };

        let cloned_state = Arc::clone(&new_app.car_rows);
        // TODO: Read from env during compilation
        thread::spawn(move || websocket_main(cloned_state, "ws://localhost:8765".to_owned()));

        new_app
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        ctx.request_repaint_after(Duration::from_millis(500));
        egui::CentralPanel::default().show(ctx, |ui| {
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
                                ui.label(
                                    egui::RichText::from(&car.license_plate_as_string)
                                        .size(TEXT_HEIGHT)
                                        .color(Color32::WHITE),
                                );
                                ui.add(egui::Button::new(
                                    egui::RichText::from("Edit")
                                        .size(50.0)
                                        .color(Color32::WHITE),
                                ));
                            });
                        });
                    });

                    ui.add_space(32.0);
                }
            });
        });
    }
}
