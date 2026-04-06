class RuleBasedFallPlugin:
    def __init__(self, confidence_threshold=0.70, aspect_ratio_threshold=1.2):
        print("⚡ Initializing Fall Detection Plugin (Rule-Based Reflex)...")
        self.conf_threshold = confidence_threshold
        self.aspect_ratio_threshold = aspect_ratio_threshold

    def process_batch(self, scene_state):
        """
        Consumes the scene_state dictionary from the Shared Backbone.
        Returns a list of alerts if a fall is mathematically detected.
        """
        alerts = []

        # Iterate through every camera currently sending data
        for cam_id, data in scene_state.items():
            
            # Iterate through every person tracked in that camera
            for track_id, person in data["tracked_people"].items():
                
                # Unpack the bounding box
                x1, y1, x2, y2 = person["bbox"]
                conf = person["confidence"]

                # Calculate geometry
                width = x2 - x1
                height = y2 - y1

                # --------------------------------------------------
                # THE REFLEX RULE
                # If width is greater than height by our threshold (e.g., 20%)
                # --------------------------------------------------
                if width > (height * self.aspect_ratio_threshold):
                    
                    # Double-check confidence so we don't trigger on a false positive shadow
                    if conf > self.conf_threshold:
                        print(f"⚠️  Fall detected on {cam_id} for track {track_id} with confidence {conf:.2f}")
                        alerts.append({
                            "cameraId": cam_id,
                            "eventType": "fall",
                            "confidence": conf
                        })

        return alerts