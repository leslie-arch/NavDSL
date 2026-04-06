class ObservationAdapter:
    def __init__(self, visual_encoder=None):
        self.encoder = visual_encoder

    def to_model_input(self, obs, instruction):
        rgb = obs["rgb"]

        if self.encoder:
            feat = self.encoder(rgb)
            return f"""
            Instruction: {instruction}
            Visual feature: {feat}
            """
        else:
            return {
                "image": rgb,
                "text": instruction
            }
