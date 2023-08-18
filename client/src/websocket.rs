use std::time::Duration;

use anyhow::anyhow;
use egui_extras::RetainedImage;
use websockets::{Frame, WebSocket};

use crate::models::{CarRow, CarRows, SharedWebSocketState, WebSocketStates};

#[tokio::main]
pub async fn websocket_main(
    car_rows: CarRows,
    websocket_state: SharedWebSocketState,
    websocket_url: &str,
) {
    loop {
        websocket_state.lock().unwrap().value = WebSocketStates::Reconnecting;
        let Ok(mut ws) = WebSocket::connect(websocket_url).await else {
            websocket_state.lock().unwrap().value = WebSocketStates::Closed("Unable to connect to websocket. Retrying in 5 seconds".to_string());
            tokio::time::sleep(Duration::from_secs(5)).await;
            continue;
        };

        match ws.receive().await {
            Ok(Frame::Text { payload, .. }) if payload == "echo" => {
                websocket_state.lock().unwrap().value = WebSocketStates::Connected;
            }
            s => {
                websocket_state.lock().unwrap().value = WebSocketStates::Closed(
                    format!("First message received was supposed to be echo, but got \"{:?}\" intead. Retrying in 5 seconds...", s)
                );
                _ = ws.close(None).await;
                tokio::time::sleep(Duration::from_secs(5)).await;
                continue;
            }
        };

        loop {
            match websocket_receive_car(&mut ws).await {
                Ok(Some(s)) => car_rows.lock().unwrap().insert(0, s),
                Ok(None) => {}
                Err(s) => {
                    websocket_state.lock().unwrap().value = WebSocketStates::Closed(format!(
                        "Error while receiving car: {}. Refreshing connection in 1 second...",
                        s
                    ));
                    _ = ws.close(None).await;
                    tokio::time::sleep(Duration::from_secs(1)).await;
                    break;
                }
            }
        }
    }
}

async fn websocket_receive_car(ws: &mut WebSocket) -> anyhow::Result<Option<CarRow>> {
    if let Frame::Binary { payload, .. } = ws.receive().await? {
        let Ok(car_image) = RetainedImage::from_image_bytes("car_image", &payload) else { 
            return Err(anyhow!("Unable to parse car image from binary representation."));
        };
        if let Frame::Binary { payload, .. } = ws.receive().await? {
            let Ok(lp_image) = RetainedImage::from_image_bytes("lp_image", &payload) else {
                return Err(anyhow!("Unable to parse license plate image from binary representation."));
            };
            if let Frame::Text { payload, .. } = ws.receive().await? {
                let lp_formatted_string: Vec<&str> = payload.split("=>").collect();
                if let (Some(lp_as_string), Some(lp_uuid)) = (
                    lp_formatted_string.first().map(|s| s.trim()),
                    lp_formatted_string
                        .get(1)
                        .map(|s| s.trim())
                        .and_then(|s| uuid::Uuid::parse_str(s).ok()),
                ) {
                    return Ok(Some(CarRow::new(
                        lp_uuid,
                        car_image,
                        lp_image,
                        lp_as_string.to_string(),
                    )));
                };
            }
        }
    }
    Ok(None)
}
