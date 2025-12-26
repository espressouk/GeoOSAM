"""
SAM3 CLIP Tokenizer Fix
Monkey-patch to fix Ultralytics SAM3 text prompt bug

Bug: Ultralytics tries to call SimpleTokenizer instance as a function
Fix: Use clip.tokenize() function instead

Author: GeoOSAM Contributors
Date: 2025-12-26
Issue: https://github.com/ultralytics/ultralytics/issues/22647
"""

import sys


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
        # Check if CLIP is available
        try:
            import clip
        except ImportError:
            print("⚠️  CLIP not installed - SAM3 text prompts unavailable")
            return False

        # Check if Ultralytics SAM3 is available
        try:
            from ultralytics.models.sam.sam3.text_encoder_ve import VETextEncoder
        except ImportError:
            print("⚠️  Ultralytics SAM3 not available - fix not needed")
            return False

        # Apply the fix
        print("🔧 Applying SAM3 CLIP tokenizer fix...")

        # Save reference to original forward method
        _original_forward = VETextEncoder.forward

        def _fixed_forward(self, text, input_boxes=None):
            """
            Fixed forward method that uses clip.tokenize() function
            instead of calling SimpleTokenizer instance.

            Original bug:
                tokenized = self.tokenizer(text, ...)  # ❌ SimpleTokenizer not callable

            Fixed:
                tokenized = clip.tokenize(text, ...)   # ✅ Use CLIP's tokenize function
            """
            if isinstance(text[0], str):
                # Process raw text strings
                assert input_boxes is None or len(input_boxes) == 0, "not supported"

                # FIX: Use clip.tokenize() function instead of self.tokenizer()
                # This is the ONE line that fixes the bug!
                tokenized = clip.tokenize(
                    text,
                    context_length=self.context_length,
                    truncate=True  # Avoid errors with long text
                ).to(self.resizer.weight.device)

                # Rest of the original code (unchanged)
                text_attention_mask = (tokenized != 0).bool()

                # Manually embed the tokens
                inputs_embeds = self.encoder.token_embedding(tokenized)
                _, text_memory = self.encoder(tokenized)

                assert text_memory.shape[1] == inputs_embeds.shape[1]

                # Invert attention mask (opposite convention in pytorch transformer)
                text_attention_mask = text_attention_mask.ne(1)

                # Transpose memory (pytorch's attention expects sequence first)
                text_memory = text_memory.transpose(0, 1)

                # Resize encoder hidden states to match decoder d_model
                text_memory_resized = self.resizer(text_memory)
            else:
                # Text already encoded, use as-is
                text_attention_mask, text_memory_resized, tokenized = text
                inputs_embeds = tokenized["inputs_embeds"]

            # Return in pytorch's convention (sequence first)
            return (
                text_attention_mask,
                text_memory_resized,
                inputs_embeds.transpose(0, 1),
            )

        # Apply the monkey-patch
        VETextEncoder.forward = _fixed_forward

        print("✅ SAM3 CLIP tokenizer fix applied successfully!")
        print("   Text prompts and exemplar mode should now work")
        return True

    except Exception as e:
        print(f"❌ Failed to apply SAM3 CLIP fix: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_sam3_text_available():
    """
    Check if SAM3 text prompts are available (CLIP + Ultralytics installed).

    Returns:
        bool: True if SAM3 text features can work, False otherwise
    """
    try:
        import clip
        from ultralytics.models.sam.sam3.text_encoder_ve import VETextEncoder
        return True
    except ImportError:
        return False


# Auto-apply fix on import (optional)
# Uncomment to apply fix automatically when this module is imported
# apply_sam3_clip_fix()
