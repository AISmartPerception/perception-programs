import importlib

from transformers import LogitsProcessor


class ThinkClosingLogitsProcessor(LogitsProcessor):
    """
    Logits processor that forces </think> token after a certain number of generated tokens
    for sequences that haven't finished yet. Once </think> is forced, stops interfering.
    """
    def __init__(self, think_close_token_id, eos_token_id, max_tokens_before_closing,
                 verbose=False):
        """
        Args:
            think_close_token_id: Token ID for </think> (must be single token)
            eos_token_id: EOS token ID to detect finished sequences
            max_tokens_before_closing: Number of tokens to generate before forcing </think>
            prompt_lengths: List of prompt lengths for each sequence in batch
            verbose: Whether to print debug info
        """
        assert isinstance(think_close_token_id, int), "</think> must be a single token"

        self.think_close_token_id = think_close_token_id
        self.eos_token_id = eos_token_id
        self.max_tokens_before_closing = max_tokens_before_closing
        self.verbose = verbose

        # Track sequences that don't need forcing (already output </think> or ended)
        self.no_forcing_needed = set()

    def __call__(self, input_ids, scores):
        """
        Manipulate logits to force </think> token after max_tokens_before_closing.

        Args:
            input_ids: Tensor of shape (batch_size, sequence_length)
            scores: Tensor of shape (batch_size, vocab_size) - the logits

        Returns:
            Modified scores
        """
        batch_size = input_ids.shape[0]
        current_length = input_ids.shape[1]

        if current_length == 0:
            return scores

        # Get last token for all sequences in batch
        last_tokens = input_ids[:, -1]

        for batch_idx in range(batch_size):
            # Skip sequences that don't need forcing
            if batch_idx in self.no_forcing_needed:
                continue

            last_token = last_tokens[batch_idx].item()

            # If sequence naturally output </think> or ended, mark as no forcing needed
            if last_token == self.think_close_token_id or last_token == self.eos_token_id:
                self.no_forcing_needed.add(batch_idx)
                continue

            # If we've reached the threshold and haven't seen </think>, force it
            if current_length == self.max_tokens_before_closing:
                scores[batch_idx, :] = -float('inf')
                scores[batch_idx, self.think_close_token_id] = 1000.0
                if self.verbose:
                    print(f"Batch {batch_idx}: Forcing </think> at token position {current_length}")

        return scores


def batch_chat_multi_image(model, tokenizer, pixel_values, questions, num_patches_lists, generation_config,
                           IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>',
                           IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
                           think_closing_tokens=512,
                           verbose=False):
    """
    Custom batch chat that supports multiple images per question for InternVL models.

    InternVL's built-in batch_chat only supports ONE <image> token per question because it only
    replaces '<image>' once per question. This custom implementation mimics the behavior of
    chat() (which loops through num_patches_list to replace multiple <image> tokens) but applies
    it across a batch of questions for parallel inference.

    Args:
        model: The InternVL model
        tokenizer: The tokenizer
        pixel_values: Concatenated pixel values for all images across all samples [total_images, C, H, W]
        questions: List of questions, one per sample in batch
        num_patches_lists: List of lists, where each inner list contains patches for each image in that sample
                          e.g., [[patches_s1_img1, patches_s1_img2], [patches_s2_img1, patches_s2_img2]]
        generation_config: Generation configuration dict
        IMG_START_TOKEN: Image start token (default: '<img>')
        IMG_END_TOKEN: Image end token (default: '</img>')
        IMG_CONTEXT_TOKEN: Image context token (default: '<IMG_CONTEXT>')
        think_closing_tokens: Number of tokens to generate before forcing </think> (default: 512)
        verbose: Whether to print debug information (default: False)

    Returns:
        List of response strings, one per question
    """
    # Import get_conv_template from the model's module dynamically
    try:
        model_module = model.__class__.__module__
        module_parts = model_module.rsplit('.', 1)[0]
        conv_module = importlib.import_module(f"{module_parts}.conversation")
        get_conv_template = conv_module.get_conv_template
    except Exception:
        # Fallback: use the model's conv_template attribute directly
        def get_conv_template(template_name):
            return model.conv_template

    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    model.img_context_token_id = img_context_token_id

    if verbose and pixel_values is not None:
        image_bs = pixel_values.shape[0]
        print(f'dynamic ViT batch size: {image_bs}')

    # Build queries for each sample in batch
    queries = []

    for idx, (question, num_patches_list) in enumerate(zip(questions, num_patches_lists)):
        # Add <image> token if not present
        if pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question

        # Get conversation template
        template = get_conv_template(model.template)
        template.system_message = model.system_message
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        # Replace each <image> token with the corresponding image tokens
        # This handles multiple images per question
        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * model.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)

        queries.append(query)

    # Tokenize all queries with left padding for batch processing
    tokenizer.padding_side = 'left'
    model_inputs = tokenizer(queries, return_tensors='pt', padding=True)
    input_ids = model_inputs['input_ids'].to(model.device)
    attention_mask = model_inputs['attention_mask'].to(model.device)

    # Get EOS token
    template = get_conv_template(model.template)
    eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())
    generation_config['eos_token_id'] = eos_token_id

    # Get </think> token ID (assert it's a single token)
    think_close_token_ids = tokenizer.encode('</think>', add_special_tokens=False)
    assert len(think_close_token_ids) == 1, f"</think> must be a single token, got {len(think_close_token_ids)} tokens"
    think_close_token_id = think_close_token_ids[0]

    # Create logits processor for forcing </think> token
    think_processor = ThinkClosingLogitsProcessor(
        think_close_token_id=think_close_token_id,
        eos_token_id=eos_token_id,
        max_tokens_before_closing=think_closing_tokens,
        verbose=verbose
    )

    # Add logits processor to generation config
    gen_config = generation_config.copy()
    logits_processor_list = gen_config.get('logits_processor', [])
    if not isinstance(logits_processor_list, list):
        logits_processor_list = []
    logits_processor_list.append(think_processor)
    gen_config['logits_processor'] = logits_processor_list

    # Generate responses with logits processor
    generation_output = model.generate(
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attention_mask,
        **gen_config
    )

    # Decode final responses
    responses = tokenizer.batch_decode(generation_output, skip_special_tokens=True)
    responses = [response.split(template.sep.strip())[0].strip() for response in responses]

    return responses
