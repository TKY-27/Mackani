use super::loading::rotary_width_from_header;
use super::*;
use serde_json::json;

#[test]
fn rotary_width_checks_architecture_and_head_width_itself() {
    assert_eq!(
        rotary_width_from_header(&json!({"arch": "qwen3_5", "head_dim": 256})).unwrap(),
        64
    );
    for (header, expected) in [
        (
            json!({"arch": "other", "head_dim": 256}),
            "main reference requires arch=qwen3_5",
        ),
        (
            json!({"arch": "qwen3_5", "head_dim": 128}),
            "qwen3_5 requires model.head_dim=256",
        ),
    ] {
        assert!(
            matches!(rotary_width_from_header(&header), Err(Error::InvalidModel(message)) if message == expected)
        );
    }
}
