# IoT Operation Suggestion Model

This project is based on user smart home operation history, containing anonymized real user operation data, and uses the TRL library for model training.

## Training Process Overview

This project corresponds to the **Post-training** stage in the paper. The complete model training process includes the following two stages:

### Pre-training

> ⚠️ **Note**: The data and code for the Pre-training stage are not provided in this version. They will be completed in future updates.

In the pre-training stage, we pre-train the base LLM on user historical operation sequences H = {O₁, O₂, ..., Oₗ}. Each operation instance Oᵢ is serialized into a text sequence containing time Oᵢᵗⁱᵐᵉ and environment Oᵢᵉⁿᵛ, followed by the corresponding action Oᵢᵃᶜᵗ = {Oᵢ,₁ᵃᶜᵗ, ..., Oᵢ,ₐᵢᵃᶜᵗ}.

The training method reframes the task as **next-action prediction**, which naturally aligns with the LLM training paradigm. We concatenate the user's historical actions O<ᵢ, current time Oᵢᵗⁱᵐᵉ, and environment information Oᵢᵉⁿᵛ as conditional input, and let the model predict the user's next action. Optimization is performed by minimizing the negative log-likelihood loss:

$$\mathcal{L}_{pt} = -\sum_{O_i \in H} \log P\left(O_i^{\text{act}} \mid O_{<i}, O_i^{\text{time}}, O_i^{\text{env}}\right)$$

By learning to predict actions from context, the model internalizes the logic of device operations. For example, the model learns device-specific functions (e.g., lights can only be turned on or off, not set to 25 degrees) and valid value spaces (e.g., air conditioner temperature must be a reasonable numerical range).

To avoid catastrophic forgetting and preserve the model's general knowledge, we mix the operation pre-training corpus with general domain corpora (such as ShareGPT, WuDao) at a 1:1 ratio to create a balanced training dataset.

### Post-training ✅ Provided in this project

After pre-training, the model has acquired the ability to understand the IoT operation domain, but two key issues remain:
1. The model struggles to apply operation knowledge to recommendation scenarios, such as suggesting operations for user-available devices
2. The model cannot generate user-friendly natural language descriptions (e.g., "turn on the air conditioner"), but can only output raw structured operations (e.g., `[Air Conditioner][Switch][On]`)

#### Fine-tuning

To address these issues, we fine-tune the LLM on specialized training corpora. Each training instance contains:
- **Operation recommendation prompt P**: Composed of user operation history O<ₗ, current time Oₗᵗⁱᵐᵉ, environment information Oₗᵉⁿᵛ, and candidate device list C
- **Ground truth action**: The operation actually performed by the user
- **Text description**: The corresponding natural language description

We use **LoRA** (Low-Rank Adaptation) for efficient fine-tuning. LoRA introduces low-rank matrices in the pre-trained LLM for incremental parameter updates, training only a small number of learnable parameters.

**Training Strategy**: We adopt an **action-first** strategy where the model first predicts the action, then generates the description. This establishes a learning flow from concrete to abstract:
- Predicting structured precise actions as the core task provides a clear "anchor" for subsequent text generation
- Once the action is determined, generating the description becomes a simpler summarization task

The loss function decomposes as:

$$\mathcal{L}_{ad} = -(\log P(O_l^{act} | P) + \log P(O_l^{desc} | P, O_l^{act}))$$

The **SFT** and **DPO** training provided in this project are designed to solve these problems, enabling the model to:
- Generate personalized recommendations based on the user's device list and historical behavior
- Output structured operation instructions for system execution

## Project Structure

```
iot_operation_suggest/
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── corpus/                   # Data directory
│   ├── train_data.jsonl        # Merged training data
│   └── eval_data.jsonl         # Merged evaluation data
├── src/                      # Source code directory
│   ├── __init__.py
│   ├── data/                 # Data processing module
│   │   ├── __init__.py
│   │   ├── config.py         # Template configuration
│   │   ├── transform.py      # Data transformation logic
│   │   └── dataset_builder.py # Dataset building (random sampling)
│   ├── training/             # Training module
│   │   ├── __init__.py
│   │   ├── sft_trainer.py    # SFT training script
│   │   └── dpo_trainer.py    # DPO training script
│   └── evaluation/           # Evaluation module
│       ├── __init__.py
│       └── evaluator.py      # Model evaluation script
├── scripts/                  # Run scripts
    ├── run_sft.sh            # SFT training launch script
    ├── run_dpo.sh            # DPO training launch script
    └── run_eval.sh           # Standalone evaluation launch script

```

## Features

### 1. Data Processing

The data processing module is responsible for converting raw user smart home operation history data into the format required for model training.

**Core Features:**
- Parse user historical operation intent sequences
- Generate device list and status information
- Build prompt and label
- **Random sampling from merged single data file**
- **Automatic validation set sampling**

### 2. SFT Training

Uses the TRL library for Supervised Fine-Tuning based on user historical behavior to predict device operations. Supports both full fine-tuning and LoRA efficient fine-tuning modes.

**Input Format:**
- `prompt`: User historical behavior + current environment info + candidate device list
- `response`: Expected device operation

**LoRA Configuration:**
- Enable LoRA fine-tuning via `--use_lora` parameter
- Configurable LoRA rank, alpha, dropout parameters
- Default target modules: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`

### 3. DPO Training

Uses the TRL library for Direct Preference Optimization, improving model performance through positive and negative sample comparison. **LoRA fine-tuning is enabled by default**.

**Sample Construction:**
- `prompt`: User historical behavior + current environment info + candidate device list
- `chosen`: Correct device operation (post_text)
- `rejected`: Online model response (online_response)

**LoRA Configuration:**
- DPO training enables LoRA by default (can be disabled via `USE_LORA=false`)
- Configurable LoRA rank, alpha, dropout parameters
- Default target modules: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`

### 4. Model Evaluation

The evaluation module supports:
- Random sampling from evaluation data file
- evaluation metrics (exact match)
- Comparison with online model
- Save prediction results

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run SFT Training

```bash
# Use default parameters (full fine-tuning, use all data)
bash scripts/run_sft.sh

# Use LoRA fine-tuning (recommended)
USE_LORA=true bash scripts/run_sft.sh

# Use LoRA + random sampling 10000 training samples
USE_LORA=true TRAIN_MAX_SAMPLES=10000 bash scripts/run_sft.sh

# Custom LoRA parameters
USE_LORA=true LORA_R=32 LORA_ALPHA=64 TRAIN_MAX_SAMPLES=5000 bash scripts/run_sft.sh
```

### 3. Run DPO Training

```bash
# Use default parameters (LoRA fine-tuning, use all data)
bash scripts/run_dpo.sh

# Random sampling 10000 training samples
TRAIN_MAX_SAMPLES=10000 bash scripts/run_dpo.sh

# Custom LoRA parameters
LORA_R=32 LORA_ALPHA=64 TRAIN_MAX_SAMPLES=5000 bash scripts/run_dpo.sh

# Disable LoRA (full fine-tuning)
USE_LORA=false bash scripts/run_dpo.sh
```

### 4. Run Standalone Evaluation

```bash
# Evaluate trained model
MODEL_PATH=./outputs/sft MAX_SAMPLES=1000 bash scripts/run_eval.sh
```

## Data Format

### Data

> ⚠️ **Note**: The data are not provided in this version. They will be completed in future updates.


### Data Schema

Each line in the JSONL file contains one user sample with the following structure:

```json
{
  "uid": "",
  "city_info": "浙江宁波",
  "device_list": [
    {
      "category": "电热毯",
      "city": "宁波",
      "country": "中国",
      "device_name": "环鼎智能水暖毯HD",
      "did": "",
      "operation_type": "mix_operation",
      "province": "浙江",
      "room_name": "卧室"
    }
  ],
  "history_intents": [
    {
      "action_time": 1760862118,
      "date_info": "20251019,节假日,1",
      "logs": [
        {
          "category": "风扇",
          "device_name": "米家直流变频台式循环扇",
          "did": "",
          "elem_type": "action",
          "field_name": "设置左右摆风",
          "formatted_time": "2025-10-19 16:21:58",
          "nice_client": "appgateway",
          "room_name": "卧室",
          "service_name": "风扇",
          "timestamp": 1760862118,
          "value": "打开",
          "value_norm": ""
        }
      ]
    }
  ],
  "actual_intents": [
    {
      "action_time": 1762097090,
      "date_info": "20251102,节假日,1",
      "logs": [
        {
          "category": "电热毯",
          "device_name": "环鼎智能水暖毯HD",
          "did": "",
          "elem_type": "device",
          "field_name": "",
          "formatted_time": "2025-11-02 23:24:50",
          "nice_client": "",
          "room_name": "卧室",
          "service_name": "",
          "timestamp": 1762097090,
          "value": "关闭",
          "value_norm": ""
        },
        {
          "category": "电热毯",
          "device_name": "环鼎智能水暖毯HD",
          "did": "",
          "elem_type": "action",
          "field_name": "调整温度",
          "formatted_time": "2025-11-02 23:24:58",
          "nice_client": "appgateway",
          "room_name": "卧室",
          "service_name": "电热毯",
          "timestamp": 1762097098,
          "value": "36",
          "value_norm": ""
        },
        {
          "category": "电热毯",
          "device_name": "环鼎智能水暖毯HD",
          "did": "",
          "elem_type": "action",
          "field_name": "设备开关",
          "formatted_time": "2025-11-02 23:24:50",
          "nice_client": "deviceshadow",
          "room_name": "卧室",
          "service_name": "电热毯",
          "timestamp": 1762097090,
          "value": "打开",
          "value_norm": ""
        }
      ]
    }
  ],
  "weather_info": {
    "aqi": "良",
    "humidity": "76",
    "sunriseTime": "2025-11-02T06:11:00+08:00",
    "sunsetTime": "2025-11-02T17:10:00+08:00",
    "temperature": "16",
    "weather": "阴转多云"
  },
  "online_response": "设备：1，动作：[电热毯][设备开关][打开]"
}
```

### Field Descriptions

#### Top-level Fields

| Field | Type | Description                                                                   |
|-------|------|-------------------------------------------------------------------------------|
| `uid` | string | user ID empty                                                                  |
| `city_info` | string | User's city location, format: `<province><city>`                              |
| `device_list` | array | List of smart home devices available to the user                              |
| `history_intents` | array | User's historical operation intents (context for prediction)                  |
| `actual_intents` | array | Current operation intents (ground truth labels for training)                  |
| `weather_info` | object | Current weather information at the time of operation                          |
| `online_response` | string | Online model's prediction response (used for DPO training as rejected sample) |

#### Device Object (`device_list[*]`)

| Field | Type | Description                                        |
|-------|------|----------------------------------------------------|
| `category` | string | Device category (e.g., "电热毯", "风扇", "空调")          |
| `city` | string | City where the device is located                   |
| `country` | string | Country where the device is located                |
| `device_name` | string | Full device name/model                             |
| `did` | string | device ID empty                                         |
| `operation_type` | string | Operation type (e.g., "mix_operation")             |
| `province` | string | Province where the device is located               |
| `room_name` | string | Room where the device is placed (e.g., "卧室", "客厅") |

#### Intent Object (`history_intents[*]` / `actual_intents[*]`)

| Field | Type | Description |
|-------|------|-------------|
| `action_time` | integer | Unix timestamp of the action |
| `date_info` | string | Date information, format: `<YYYYMMDD>,<holiday_type>,<is_holiday>` |
| `logs` | array | Detailed operation logs within this intent |

#### Log Object (`logs[*]`)

| Field | Type | Description                                                                                                                                |
|-------|------|--------------------------------------------------------------------------------------------------------------------------------------------|
| `category` | string | Device category                                                                                                                            |
| `device_name` | string | Full device name/model                                                                                                                     |
| `did` | string | device ID empty                                                                                                                                 |
| `elem_type` | string | Element type: `"device"` (device state) or `"action"` (user action) or `"env"` (environment information) or `"event"` (events information) |
| `field_name` | string | Operation field name (e.g., "设备开关", "调整温度", "设置左右摆风")                                                                                      |
| `formatted_time` | string | Human-readable timestamp, format: `YYYY-MM-DD HH:MM:SS`                                                                                    |
| `nice_client` | string | Client source (e.g., "appgateway" (app operation), "deviceshadow" (accept recommend), "miioproc-openhome" (voice operation))               |
| `room_name` | string | Room where the device is located                                                                                                           |
| `service_name` | string | Service name (usually same as category)                                                                                                    |
| `timestamp` | integer | Unix timestamp of the log entry                                                                                                            |
| `value` | string | Operation value (e.g., "打开", "关闭", "36")                                                                                                   |
| `value_norm` | string | Normalized value (optional)                                                                                                                |

#### Weather Object (`weather_info`)

| Field | Type | Description |
|-------|------|-------------|
| `aqi` | string | Air Quality Index level (e.g., "优", "良", "轻度污染") |
| `humidity` | string | Relative humidity percentage |
| `sunriseTime` | string | Sunrise time in ISO 8601 format |
| `sunsetTime` | string | Sunset time in ISO 8601 format |
| `temperature` | string | Current temperature in Celsius |
| `weather` | string | Weather condition description (e.g., "阴转多云", "晴", "小雨") |

### Key Concepts

- **`history_intents` vs `actual_intents`**: History intents serve as context for the model to understand user behavior patterns. Actual intents are the target operations the model should predict.
- **`online_response`**: Used in DPO training as the "rejected" sample, representing what the current online model would predict. This helps the model learn to improve upon baseline predictions.

### Output Format

After training, the output directory will contain:
- Model files
- `eval_results/metrics.json`: Evaluation metrics
- `eval_results/predictions.jsonl`: Prediction results

## Configuration

### Training Data Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `TRAIN_DATA_FILE` | Training data file | `corpus/train_data.jsonl` |
| `TRAIN_MAX_SAMPLES` | Max training samples | `-1` (all) |
| `VAL_RATIO` | Validation set ratio | `0.1` |
| `VAL_MAX_SAMPLES` | Max validation samples | `10000` |

### Evaluation Data Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `EVAL_DATA_FILE` | Evaluation data file | `corpus/eval_data.jsonl` |
| `EVAL_MAX_SAMPLES` | Max evaluation samples | `5000` |
| `MAX_SAMPLES` | Max samples for standalone eval | `500` |

### LoRA Configuration (SFT/DPO Training)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `USE_LORA` | Whether to use LoRA fine-tuning | `false` |
| `LORA_R` | LoRA rank | `16` |
| `LORA_ALPHA` | LoRA alpha parameter | `32` |
| `LORA_DROPOUT` | LoRA dropout ratio | `0.05` |
| `LORA_TARGET_MODULES` | LoRA target modules | `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` |


## Dependencies

- Python >= 3.10
- PyTorch >= 2.6
- Transformers >= 4.50.0
- TRL >= 0.12.0
- Jinja2 >= 3.1.6
- PEFT >= 0.7.0
