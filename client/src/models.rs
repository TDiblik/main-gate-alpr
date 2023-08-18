use std::sync::{Arc, Mutex};

use egui_extras::RetainedImage;
use uuid::Uuid;

pub type CarRows = Arc<Mutex<Vec<CarRow>>>;

pub struct CarRow {
    pub uuid: Uuid,
    pub car_image: RetainedImage,
    pub license_plate_image: RetainedImage,
    pub license_plate_as_string: String,
    pub received_at_formatted: String,
}

impl CarRow {
    pub fn new(
        uuid: Uuid,
        car_image: RetainedImage,
        license_plate_image: RetainedImage,
        license_plate_as_string: String,
    ) -> Self {
        Self {
            uuid,
            car_image,
            license_plate_image,
            license_plate_as_string,
            received_at_formatted: chrono::Local::now()
                .naive_local()
                .format("%d.%m.%Y %H:%M:%S")
                .to_string(),
        }
    }
}

pub type SharedWebSocketState = Arc<Mutex<WebSocketState>>;
pub struct WebSocketState {
    pub value: WebSocketStates,
}
impl Default for WebSocketState {
    fn default() -> Self {
        Self {
            value: WebSocketStates::Closed("Didn't even try opening (yet).".to_string()),
        }
    }
}
pub enum WebSocketStates {
    Connected,
    Reconnecting,
    Closed(String),
}
