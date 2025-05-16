import os

from typing import List, Optional, Tuple
from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams, TokensPrompt, LLM
from vllm.lora.request import LoRARequest


PROMPT_TEMPLATE_FACTS = "Based on the given text, give all the mentioned atomic fact sentences, one per line. Each sentence should be decontextualized with resolved pronouns (eg. don't use 'this' or 'that', mention the actual object) and self-explanatory without any additional context. text: "

class LLMEvaluator:
    def __init__(self, 
                 base_model: str = "meta-llama/Llama-3.1-8B-Instruct",
                 fact_generation_lora_path: str = "algoprog/fact-generation-llama-3.1-8b-instruct-lora"):
        self.base_model = base_model
        self.fact_generation_lora_path = fact_generation_lora_path
        self._initialize_engine()

    def _initialize_engine(self):
        # engine_args = EngineArgs(model=self.base_model,
        #     dtype="half",
        #     max_model_len=4096,
        #     enable_lora=True,
        #     max_loras=1,
        #     max_lora_rank=64,  # Matches your rank=64 from training
        #     max_cpu_loras=2,
        #     max_num_seqs=8,
        #     download_dir="/gypsum/work1/zamani/asalemi/RAG_VS_LoRA_Personalization/cache")
        # self.engine = LLMEngine.from_engine_args(engine_args)
        self.engine = LLM(
            self.base_model, 
            enable_lora=True,
            max_lora_rank=64,
            max_loras=1,
            max_cpu_loras=2,
            max_model_len=4096,
            download_dir="/gypsum/work1/zamani/asalemi/RAG_VS_LoRA_Personalization/cache"
        )
        self.tokenizer = self.engine.get_tokenizer()
        

    def _build_prompt(self, text: str) -> str:
        input_ids = self.tokenizer.apply_chat_template(
            [{'role': 'user', 'content': text}],
            add_generation_prompt=True
            )
        return TokensPrompt(prompt_token_ids=input_ids)

    def generate_facts(self, texts: List[str]) -> List[List[str]]:
        results = []
        request_id = 0
        # pending_texts = texts.copy()

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=2048,
        )

        ######## ME START
        prompts = []
        for text in texts:
            prompt = self._build_prompt(PROMPT_TEMPLATE_FACTS + text)
            prompts.append(prompt)
        lora_request = LoRARequest(
            "fact-generator-lora",  # Adapter name
            1,  # Adapter ID
            self.fact_generation_lora_path
        )
        outputs = self.engine.generate(prompts, sampling_params, lora_request=lora_request)
        for output in outputs:
            generated_text = output.outputs[0].text
            facts = generated_text.split('\n')
            facts = [fact.strip() for fact in facts if fact.strip()]
            results.append(facts)
        # while self.engine.has_unfinished_requests():
        #     request_outputs: List[RequestOutput] = self.engine.step()
        #     for request_output in request_outputs:
        #         if request_output.finished:
        #             generated_text = request_output.outputs[0].text
        #             facts = generated_text.split('\n')
        #             facts = [fact.strip() for fact in facts if fact.strip()]
        #             results.append(facts)
        ####### ME END
        # while pending_texts or self.engine.has_unfinished_requests():
        #     if pending_texts:
        #         text = pending_texts.pop(0)
        #         prompt = self._build_prompt(PROMPT_TEMPLATE_FACTS + text)
                
        #         lora_request = LoRARequest(
        #             "fact-generator-lora",  # Adapter name
        #             1,  # Adapter ID
        #             self.fact_generation_lora_path
        #         )
        #         self.engine.add_request(
        #             str(request_id),
        #             prompt,
        #             sampling_params,
        #             lora_request=lora_request
        #         )
        #         request_id += 1

            # request_outputs: List[RequestOutput] = self.engine.step()

            # for request_output in request_outputs:
            #     if request_output.finished:
            #         generated_text = request_output.outputs[0].text
            #         facts = generated_text.split('\n')
            #         facts = [fact.strip() for fact in facts if fact.strip()]
            #         results.append(facts)

        return results

    def generate(self, texts: List[str]) -> List[str]:
        results = []
        request_id = 0
        # pending_texts = texts.copy()

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=2048,
        )

        ######## ME START
        prompts = []
        for text in texts:
            prompt = self._build_prompt(text)
            prompts.append(prompt)
        outputs = self.engine.generate(prompts, sampling_params)
        for output in outputs:
            generated_text = output.outputs[0].text
            results.append(generated_text)

        # while self.engine.has_unfinished_requests():
        #     request_outputs: List[RequestOutput] = self.engine.step()
        #     for request_output in request_outputs:
        #         if request_output.finished:
        #             generated_text = request_output.outputs[0].text
        #             results.append(generated_text)
        ####### ME END

        # while pending_texts or self.engine.has_unfinished_requests():
        #     if pending_texts:
        #         prompt = self._build_prompt(pending_texts.pop(0))
        #         self.engine.add_request(
        #             str(request_id),
        #             prompt,
        #             sampling_params
        #         )
        #         request_id += 1

            # request_outputs: List[RequestOutput] = self.engine.step()

            # for request_output in request_outputs:
            #     if request_output.finished:
            #         generated_text = request_output.outputs[0].text
            #         results.append(generated_text)

        return results

if __name__ == "__main__":
    llm_evaluator = LLMEvaluator()
    print("generating response...")
    print(llm_evaluator.generate(["What is the capital of France?"]))

    print("generating facts...")
    print(llm_evaluator.generate_facts(["The quick brown fox jumps over the lazy dog. The dog is a good dog."]))