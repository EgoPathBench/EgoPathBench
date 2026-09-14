"""Compatibility shims for running local LLaMA-Factory with newer Transformers."""

from __future__ import annotations

import builtins
import importlib.util
import sys


_ORIGINAL_IMPORT = builtins.__import__


def _ensure_vision2seq_alias(module):
    if (
        not hasattr(module, "AutoModelForVision2Seq")
        and hasattr(module, "AutoModelForImageTextToText")
    ):
        module.AutoModelForVision2Seq = module.AutoModelForImageTextToText


def _ensure_utils_aliases(module):
    if not hasattr(module, "is_torch_sdpa_available"):
        def is_torch_sdpa_available():
            try:
                import torch

                return hasattr(torch.nn.functional, "scaled_dot_product_attention")
            except Exception:
                return False

        module.is_torch_sdpa_available = is_torch_sdpa_available

    if not hasattr(module, "is_safetensors_available"):
        def is_safetensors_available():
            return importlib.util.find_spec("safetensors") is not None

        module.is_safetensors_available = is_safetensors_available

    if not hasattr(module, "is_jieba_available"):
        def is_jieba_available():
            return importlib.util.find_spec("jieba") is not None

        module.is_jieba_available = is_jieba_available

    if not hasattr(module, "is_nltk_available"):
        def is_nltk_available():
            return importlib.util.find_spec("nltk") is not None

        module.is_nltk_available = is_nltk_available


def _ensure_modeling_auto_aliases(module):
    if (
        not hasattr(module, "MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES")
        and hasattr(module, "MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES")
    ):
        module.MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES = module.MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES


def _patch_hf_argparser(module):
    parser_cls = getattr(module, "HfArgumentParser", None)
    if parser_cls is None or getattr(parser_cls, "_navbench_conflict_patch", False):
        return

    original_init = parser_cls.__init__

    def patched_init(self, *args, **kwargs):
        kwargs.setdefault("conflict_handler", "resolve")
        original_init(self, *args, **kwargs)

    parser_cls.__init__ = patched_init
    parser_cls._navbench_conflict_patch = True


def _patch_training_arguments(module):
    args_cls = getattr(module, "TrainingArguments", None)
    if args_cls is None or getattr(args_cls, "_navbench_overwrite_output_dir_patch", False):
        return

    if not hasattr(args_cls, "overwrite_output_dir"):
        args_cls.overwrite_output_dir = False
    args_cls._navbench_overwrite_output_dir_patch = True


def _patch_tokenizers_backend(module):
    backend_cls = getattr(module, "TokenizersBackend", None)
    if backend_cls is None or getattr(backend_cls, "_navbench_special_tokens_patch", False):
        return

    if not hasattr(backend_cls, "additional_special_tokens_ids"):
        backend_cls.additional_special_tokens_ids = property(lambda self: [])
    if not hasattr(backend_cls, "additional_special_tokens"):
        backend_cls.additional_special_tokens = property(lambda self: [])
    backend_cls._navbench_special_tokens_patch = True


def _patch_llamafactory_sft_trainer(module):
    trainer_cls = getattr(module, "CustomSeq2SeqTrainer", None)
    if trainer_cls is None or getattr(trainer_cls, "_navbench_sampler_patch", False):
        return

    def patched_get_train_sampler(self, train_dataset=None, *args, **kwargs):
        dataset = train_dataset if train_dataset is not None else self.train_dataset
        if self.finetuning_args.disable_shuffling:
            import torch

            return torch.utils.data.SequentialSampler(dataset)

        sampler_fn = super(trainer_cls, self)._get_train_sampler
        try:
            return sampler_fn(train_dataset, *args, **kwargs)
        except TypeError:
            return sampler_fn()

    trainer_cls._get_train_sampler = patched_get_train_sampler
    trainer_cls._navbench_sampler_patch = True


def _patch_loaded_llamafactory_modules(name, module):
    candidates = []
    module_name = getattr(module, "__name__", "")
    if module_name == "llamafactory.train.sft.trainer":
        candidates.append(module)
    loaded = sys.modules.get("llamafactory.train.sft.trainer")
    if loaded is not None:
        candidates.append(loaded)

    for candidate in candidates:
        _patch_llamafactory_sft_trainer(candidate)


def _patched_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and name == "transformers" and fromlist and "AutoModelForVision2Seq" in fromlist:
        module = _ORIGINAL_IMPORT(name, globals, locals, (), level)
        _ensure_vision2seq_alias(module)
        return module

    module = _ORIGINAL_IMPORT(name, globals, locals, fromlist, level)
    if level == 0 and name == "transformers":
        _ensure_vision2seq_alias(module)
    elif level == 0 and name == "transformers.utils":
        _ensure_utils_aliases(module)
    elif level == 0 and name == "transformers.models.auto.modeling_auto":
        _ensure_modeling_auto_aliases(module)
    elif level == 0 and name == "transformers.hf_argparser":
        _patch_hf_argparser(module)
    elif level == 0 and name == "transformers.training_args":
        _patch_training_arguments(module)
    elif level == 0 and name == "transformers.tokenization_utils_tokenizers":
        _patch_tokenizers_backend(module)
    _patch_loaded_llamafactory_modules(name, module)
    return module


builtins.__import__ = _patched_import

try:
    import transformers

    _ensure_vision2seq_alias(transformers)
    import transformers.utils

    _ensure_utils_aliases(transformers.utils)
    import transformers.models.auto.modeling_auto

    _ensure_modeling_auto_aliases(transformers.models.auto.modeling_auto)
    import transformers.hf_argparser

    _patch_hf_argparser(transformers.hf_argparser)
    import transformers.training_args

    _patch_training_arguments(transformers.training_args)
    import transformers.tokenization_utils_tokenizers

    _patch_tokenizers_backend(transformers.tokenization_utils_tokenizers)
except Exception:
    pass

try:
    trainer_module = sys.modules.get("llamafactory.train.sft.trainer")
    if trainer_module is not None:
        _patch_llamafactory_sft_trainer(trainer_module)
except Exception:
    pass
