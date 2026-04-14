FROM public.ecr.aws/hpc-cloud/nccl-tests:latest

RUN apt update && apt install -y nvtop

RUN mkdir -p /wordle-grpo /checkpoints
RUN ln -s /usr/bin/python3 /usr/bin/python

# Install Miniconda (Python 3.12)
RUN mkdir -p /opt/miniconda3 \
    && curl -L https://repo.anaconda.com/miniconda/Miniconda3-py312_25.3.1-1-Linux-x86_64.sh -o /tmp/Miniconda3.sh \
    && bash /tmp/Miniconda3.sh -b -f -p /opt/miniconda3 \
    && rm /tmp/Miniconda3.sh \
    && /opt/miniconda3/bin/conda init bash
ENV PATH="/opt/miniconda3/bin:${PATH}"

# Install Rust (required by TRL's 'outlines' dependency)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

COPY src/ /wordle-grpo/
RUN --mount=type=cache,target=/root/.cache/pip pip install -r /wordle-grpo/requirements.txt

# Upgrade TRL to 0.26.2 (adds trl.experimental.openenv module required by the training script).
# Use --no-deps to avoid pulling in incompatible vllm version constraints.
RUN pip install --no-cache-dir --no-deps 'trl==0.26.2'

# Install mergekit (required by TRL callbacks -> mergekit_utils) without deps
# to avoid downgrading accelerate/safetensors.
RUN pip install --no-cache-dir --no-deps mergekit immutables==0.21

# Fix TRL bug: transformers._is_package_available() returns a tuple (bool, version)
# but TRL uses the raw result as a boolean — a non-empty tuple is always truthy.
# Wrap _is_package_available to return only the boolean component.
RUN python -c "\
import pathlib; \
p = pathlib.Path('/opt/miniconda3/lib/python3.12/site-packages/trl/import_utils.py'); \
src = p.read_text(); \
old = 'from transformers.utils.import_utils import _is_package_available'; \
new = 'from transformers.utils.import_utils import _is_package_available as _orig_is_pkg_avail\n\ndef _is_package_available(pkg_name, return_version=False):\n    result = _orig_is_pkg_avail(pkg_name, return_version=return_version)\n    if return_version:\n        return result if isinstance(result, tuple) else (result, None)\n    return result[0] if isinstance(result, tuple) else result'; \
p.write_text(src.replace(old, new))"

# Fix GRPOTrainer compatibility with transformers 5.x (warnings_issued attr)
# and vLLM 0.8.x (logprobs_mode kwarg not supported).
RUN python -c "\
import pathlib; \
p = pathlib.Path('/opt/miniconda3/lib/python3.12/site-packages/trl/trainer/grpo_trainer.py'); \
src = p.read_text(); \
src = src.replace(\
    'model.warnings_issued[\"estimate_tokens\"] = True', \
    'if hasattr(model, \"warnings_issued\"): model.warnings_issued[\"estimate_tokens\"] = True'); \
src = src.replace('logprobs_mode=\"processed_logprobs\",\n', ''); \
p.write_text(src)"

# Flash Attention (requires torch, must install separately with --no-build-isolation)
RUN pip install flash-attn>=2.5.0 --no-build-isolation

# FlashInfer for optimized inference
RUN pip install flashinfer-python -i https://flashinfer.ai/whl/cu126/torch2.6/

WORKDIR /wordle-grpo
