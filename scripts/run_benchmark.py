"""
NavBench3D - Step 7: Benchmark Evaluation Framework.

Runs VQA questions through multiple VLMs and collects predictions.

Usage:
    python scripts/run_benchmark.py --config configs/default.yaml \
        --questions data/vqa_questions/all_questions.json \
        --model gpt-4o --output results/gpt4o_predictions.json
"""

import os
import json
import time
import base64
import argparse
from pathlib import Path
from typing import Optional

import yaml


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def encode_image(image_path: str) -> str:
    """Read and base64-encode an image file."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


class VLMClient:
    """Unified VLM API client supporting multiple providers."""

    def __init__(self, model_name: str, api_type: str, config: dict):
        self.model_name = model_name
        self.api_type = api_type
        self.max_tokens = config["vlm"]["max_tokens"]
        self.temperature = config["vlm"]["temperature"]
        self._client = None

    def _get_openai_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI()
        return self._client

    def _get_anthropic_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic()
        return self._client

    def query(self, system_prompt: str, user_prompt: str,
              image_paths: list[str]) -> dict:
        """Send a query to the VLM and return the response."""
        start_time = time.time()

        try:
            if self.api_type == "openai":
                result = self._query_openai(system_prompt, user_prompt, image_paths)
            elif self.api_type == "anthropic":
                result = self._query_anthropic(system_prompt, user_prompt, image_paths)
            elif self.api_type == "google":
                result = self._query_google(system_prompt, user_prompt, image_paths)
            elif self.api_type == "dashscope":
                result = self._query_dashscope(system_prompt, user_prompt, image_paths)
            else:
                raise ValueError(f"Unsupported API type: {self.api_type}")

            result["latency"] = time.time() - start_time
            result["success"] = True
            return result

        except Exception as e:
            return {
                "output": "",
                "success": False,
                "error": str(e),
                "latency": time.time() - start_time,
            }

    def _query_openai(self, system_prompt: str, user_prompt: str,
                      image_paths: list[str]) -> dict:
        client = self._get_openai_client()

        content = []
        for img_path in image_paths:
            if Path(img_path).exists():
                b64 = encode_image(img_path)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"},
                })

        content.append({"type": "text", "text": user_prompt})

        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        return {
            "output": response.choices[0].message.content,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            },
        }

    def _query_anthropic(self, system_prompt: str, user_prompt: str,
                         image_paths: list[str]) -> dict:
        client = self._get_anthropic_client()

        content = []
        for img_path in image_paths:
            if Path(img_path).exists():
                b64 = encode_image(img_path)
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": b64,
                    },
                })

        content.append({"type": "text", "text": user_prompt})

        response = client.messages.create(
            model=self.model_name,
            system=system_prompt,
            messages=[{"role": "user", "content": content}],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        return {
            "output": response.content[0].text,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        }

    def _query_google(self, system_prompt: str, user_prompt: str,
                      image_paths: list[str]) -> dict:
        import google.generativeai as genai
        from PIL import Image

        model = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=system_prompt,
        )

        content_parts = []
        for img_path in image_paths:
            if Path(img_path).exists():
                img = Image.open(img_path)
                content_parts.append(img)

        content_parts.append(user_prompt)

        response = model.generate_content(
            content_parts,
            generation_config=genai.GenerationConfig(
                max_output_tokens=self.max_tokens,
                temperature=self.temperature,
            ),
        )

        return {
            "output": response.text,
            "usage": {},
        }

    def _query_dashscope(self, system_prompt: str, user_prompt: str,
                         image_paths: list[str]) -> dict:
        from openai import OpenAI

        client = OpenAI(
            api_key=os.environ.get("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        content = []
        for img_path in image_paths:
            if Path(img_path).exists():
                b64 = encode_image(img_path)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                })

        content.append({"type": "text", "text": user_prompt})

        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        return {
            "output": response.choices[0].message.content,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            },
        }


def run_benchmark(questions: list[dict], client: VLMClient,
                  output_path: str, resume: bool = True,
                  max_questions: Optional[int] = None):
    """Run benchmark evaluation on all questions."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Load existing results for resume
    existing = {}
    if resume and output_file.exists():
        with open(output_file) as f:
            for pred in json.load(f):
                existing[pred["question_id"]] = pred
        print(f"Resuming: found {len(existing)} existing predictions")

    results = list(existing.values())
    questions_to_run = [q for q in questions if q["question_id"] not in existing]

    if max_questions:
        questions_to_run = questions_to_run[:max_questions]

    print(f"Running {len(questions_to_run)} questions ({len(existing)} already done)")

    for i, question in enumerate(questions_to_run):
        qid = question["question_id"]
        print(f"[{i+1}/{len(questions_to_run)}] {qid} (L{question['level']})", end=" ")

        response = client.query(
            question["system_prompt"],
            question["user_prompt"],
            question.get("images", []),
        )

        prediction = {
            "question_id": qid,
            "scene_id": question["scene_id"],
            "view_id": question["view_id"],
            "level": question["level"],
            "source": question.get("source", "internscene"),
            "model": client.model_name,
            "output": response.get("output", ""),
            "success": response.get("success", False),
            "error": response.get("error"),
            "latency": response.get("latency", 0),
            "usage": response.get("usage", {}),
        }
        results.append(prediction)

        status = "OK" if response["success"] else f"ERR: {response.get('error', '')[:50]}"
        print(f"[{status}] ({response.get('latency', 0):.1f}s)")

        # Save incrementally
        if (i + 1) % 10 == 0 or i == len(questions_to_run) - 1:
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)

        # Rate limiting
        time.sleep(0.5)

    print(f"\nDone. Total predictions: {len(results)}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Run NavBench3D Benchmark")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--questions", required=True, help="Questions JSON file")
    parser.add_argument("--model", required=True, help="Model name (e.g., gpt-4o)")
    parser.add_argument("--api", default=None, help="API type (openai/anthropic/google/dashscope)")
    parser.add_argument("--output", default=None, help="Output predictions file")
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--level", type=int, default=None, help="Only run specific level")
    parser.add_argument("--source", default=None, help="Only run specific source (e.g., internscene)")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)

    # Find API type for model
    api_type = args.api
    if not api_type:
        for model_cfg in config["vlm"]["models"]:
            if model_cfg["name"] == args.model:
                api_type = model_cfg["api"]
                break
        if not api_type:
            print(f"Unknown model '{args.model}'. Specify --api explicitly.")
            return

    # Load questions
    with open(args.questions) as f:
        questions = json.load(f)

    if args.level:
        questions = [q for q in questions if q["level"] == args.level]
        print(f"Filtered to {len(questions)} Level {args.level} questions")
    if args.source:
        questions = [q for q in questions if q.get("source", "internscene") == args.source]
        print(f"Filtered to {len(questions)} source={args.source} questions")

    print(f"Total questions: {len(questions)}")
    print(f"Model: {args.model} (API: {api_type})")

    # Create client
    client = VLMClient(args.model, api_type, config)

    # Output path
    output_path = args.output or f"data/results/{args.model.replace('/', '_')}_predictions.json"

    # Run
    run_benchmark(
        questions, client, output_path,
        resume=not args.no_resume,
        max_questions=args.max_questions,
    )


if __name__ == "__main__":
    main()
