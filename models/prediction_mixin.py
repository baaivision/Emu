from peft import PeftModel
import torch


class PredictClassMixin:
    @torch.no_grad()
    def predict(
        self,
        samples,
        candidates,
        n_segments=1,
    ):
        if type(candidates[0]) in (list, tuple):
            results = []

            for i in range(samples["image"].size(0)):
                this_sample = {
                    "image": samples["image"][i].unsqueeze(0),
                }
                
                if "prompt" in samples.keys():
                    this_sample["prompt"] = [samples["prompt"][i]]

                this_result = self._predict(this_sample, candidates[i], n_segments)
                results.append(this_result)

            try:
                results = torch.cat(results, dim=0)
            except:
                results = [res.tolist()[0] for res in results]

            return results

        return self._predict(samples, candidates, n_segments)

    def _predict(
        self,
        samples,
        candidates,
        n_segments=1,
    ):
        image = samples["image"]
        if image is not None:
            image = image.to(dtype=torch.bfloat16)
            image_features = self.ln_visual(self.visual.forward_features(image))
            image_features = self.cformer(image_features).squeeze().to(dtype=torch.bfloat16)

        prompt = samples["prompt"] if "prompt" in samples.keys() else self.prompt
        from models.modeling_llama import LLaMAForClsAndRegression
        if isinstance(self.decoder, LLaMAForClsAndRegression):
            self.decoder.tokenizer.padding_side = "left"

        input_tokens = self.decoder.tokenizer(
            prompt, 
            padding="longest", 
            return_tensors="pt",
            add_special_tokens=True,
        ).to(self.args.device)

        self.decoder.tokenizer.padding_side = "right"

        input_ids = input_tokens.input_ids[0]
        encoder_atts = input_tokens.attention_mask[0]

        img_token_id = self.decoder.tokenizer.convert_tokens_to_ids(["<image>"])[0]  # 32003
        img_token_idx_list = input_ids.eq(img_token_id).squeeze() 

        if self.args.instruct:
            inputs_embeds = self.decoder.lm.model.model.embed_tokens(input_ids)
        else:
            inputs_embeds = self.decoder.lm.model.embed_tokens(input_ids)

        if image is not None:
            image_features = image_features.reshape(-1, image_features.shape[-1])
            inputs_embeds[img_token_idx_list] = image_features

        inputs_embeds = inputs_embeds.unsqueeze(0)
        encoder_atts = encoder_atts.unsqueeze(0)

        empty_targets = torch.empty_like(encoder_atts, dtype=torch.long).fill_(-100)

        n_cands = len(candidates)
        all_losses = []
        for n in range(n_segments):
            seg_len = n_cands // n_segments
            if n == (n_segments - 1):
                seg_len = n_cands - seg_len * (n_segments - 1)

            start_i = n * (n_cands // n_segments)
            end_i = start_i + seg_len

            this_output_tokens = self.decoder.tokenizer(
                candidates[start_i:end_i],
                return_tensors="pt",
                padding="longest",
                add_special_tokens=False, 
            ).to(inputs_embeds.device)

            if isinstance(self.decoder.lm, PeftModel):
                outputs_embeds = self.decoder.lm.model.model.embed_tokens(this_output_tokens.input_ids)
            else:
                outputs_embeds = self.decoder.lm.model.embed_tokens(this_output_tokens.input_ids)
            
            this_inputs_embeds = torch.cat([
                inputs_embeds.expand(seg_len, -1, -1), outputs_embeds
            ], dim=1)
            this_attention_mask = torch.cat([
                encoder_atts.expand(seg_len, -1), this_output_tokens.attention_mask
            ], dim=1)
            this_targets = this_output_tokens.input_ids.masked_fill(
                this_output_tokens.input_ids == self.decoder.tokenizer.pad_token_id, -100,
            )
            this_targets = torch.cat([
                empty_targets.expand(seg_len, -1), this_targets
            ], dim=1)

            outputs = self.decoder.lm(
                inputs_embeds=this_inputs_embeds,
                attention_mask=this_attention_mask,
                return_dict=True,
                labels=this_targets,
                reduction='none',
            )

            loss = outputs.llm_loss.reshape(1, seg_len)
            all_losses.append(loss)

        all_losses = torch.cat(all_losses, dim=-1)
        output_class_ranks = torch.argsort(all_losses, dim=-1)

        return output_class_ranks
