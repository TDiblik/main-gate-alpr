use std::sync::{Arc, Mutex};

use egui_extras::RetainedImage;
use uuid::Uuid;

pub type CarRows = Arc<Mutex<Vec<CarRow>>>;

pub struct CarRow {
    pub uuid: Uuid,
    pub car_image: RetainedImage,
    pub license_plate_image: RetainedImage,
    pub license_plate_as_string: String,
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
        }
    }
}