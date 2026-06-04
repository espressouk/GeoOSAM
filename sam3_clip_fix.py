"""
SAM3 CLIP Tokenizer Fix
Monkey-patch to fix Ultralytics SAM3 text prompt bug

Bug: Ultralytics tries to call SimpleTokenizer instance as a function
Fix: Use clip.tokenize() function instead

Author: GeoOSAM Contributors
Date: 2025-12-26
Issue: https://github.com/ultralytics/ultralytics/issues/22647
"""


def apply_sam3_clip_fix():
    """
    Apply monkey-patch to fix Ultralytics SAM3 CLIP tokenizer bug.

    This fixes the error:
        TypeError: 'SimpleTokenizer' object is not callable

    Call this BEFORE using SAM3 text prompts or exemplar mode.
    Safe to call even if SAM3 or CLIP not installed.

    Returns:
        bool: True if fix applied successfully, False otherwise
    """
    try:
        try:
            import clip
        except ImportError:
            return False

        try:
            from ultralytics.models.sam.sam3.text_encoder_ve import VETextEncoder
        except ImportError:
            return False

        def _fixed_forward(self, text, input_boxes=None):
            # Workaround for https://github.com/ultralytics/ultralytics/issues/22647:
            # SimpleTokenizer is not callable, so use clip.tokenize() directly.
            if isinstance(text[0], str):
                assert input_boxes is None or len(input_boxes) == 0, "not supported"
                tokenized = clip.tokenize(
                    text,
                    context_length=self.context_length,
                    truncate=True
                ).to(self.resizer.weight.device)
                text_attention_mask = (tokenized != 0).bool()
                inputs_embeds = self.encoder.token_embedding(tokenized)
                _, text_memory = self.encoder(tokenized)
                assert text_memory.shape[1] == inputs_embeds.shape[1]
                text_attention_mask = text_attention_mask.ne(1)
                text_memory = text_memory.transpose(0, 1)
                text_memory_resized = self.resizer(text_memory)
            else:
                text_attention_mask, text_memory_resized, tokenized = text
                inputs_embeds = tokenized["inputs_embeds"]

            return (
                text_attention_mask,
                text_memory_resized,
                inputs_embeds.transpose(0, 1),
            )

        VETextEncoder.forward = _fixed_forward
        return True

    except Exception:
        return False


def check_sam3_text_available():
    """
    Check if SAM3 text prompts are available (CLIP + Ultralytics installed).

    Returns:
        bool: True if SAM3 text features can work, False otherwise
    """
    try:
        import clip  # noqa: F401
        from ultralytics.models.sam.sam3.text_encoder_ve import VETextEncoder  # noqa: F401
        return True
    except ImportError:
        return False
