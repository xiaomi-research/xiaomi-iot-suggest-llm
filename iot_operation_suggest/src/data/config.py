#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2025 Xiaomi Corporation.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
IoT User Sequence SFT Template Configuration

This module defines template configurations for generating prompts.
"""

# Weekday mapping
WEEKDAY_MAP = {
    "1": "星期日",
    "2": "星期一",
    "3": "星期二",
    "4": "星期三",
    "5": "星期四",
    "6": "星期五",
    "7": "星期六"
}

# Environment attributes that use raw values
USE_RAW_VALUE_ENV_ATTR = ("温度", "湿度")

# Action attributes without values
NO_VALUE_ACTION_ATTR = ("开始清扫", "停止清扫", "暂停清扫", "继续清扫")

# v1_weather template configuration
IOT_USER_SEQ_SFT_PROMPT_CONFIG = {
    "instruct": "以下是用户操作家庭智能设备的历史：\n",
    
    "template": (
        "==========\n"
        "{{ history }}\n"
        "==========\n"
        "请根据以上用户历史行为、可操作的设备列表、环境事件信息，给出用户最有可能的设备操作。\n"
        "{{ final_intent }}"
    ),
    
    "single_elem": (
        "时间：\n"
        "{{ time_info }}\n"
        "环境：\n"
        "{{ env_info }}\n"
        "事件：\n"
        "{{ event_info }}\n"
        "指令：\n"
        "{{ cmd_info }}\n"
        "执行：\n"
        "{{ exec_info }}\n"
    ),
    
    "final_elem_input": (
        "可操作设备及状态：\n"
        "{{ device_list }}\n"
        "时间：\n"
        "{{ time_info }}\n"
        "地点：\n"
        "{{ city_info }}\n"
        "天气：\n"
        "{{ weather_info }}\n"
        "环境：\n"
        "{{ env_info }}\n"
        "事件：\n"
        "{{ event_info }}\n"
        "{% if iot_seq_output_cmd and iot_seq_cmd_as_guidance %}"
        "指令：\n"
        "{% else %}"
        "执行：\n"
        "{% endif %}"
    ),
    
    "single_env_info": "[{{ room_name }}][{{ attribute }}][{{ value }}]",
    
    "single_event_info": "[{{ room_name }}][{{ attribute }}]",
    
    "single_exec_info": (
        "{% if device_name is string %}"
        "{{ device_name }}"
        "{% else %}"
        "{{ device_name | join(' | ') }}"
        "{% endif %}"
        " [{{ room_name }}][{{ service_name }}][{{ field_name }}]"
    ),
    
    "single_device_info": "{{ no }}. {{ device_name }} [{{ category }}][{{ room_name }}][{{ device_state }}]",
    
    "merged_exec_info": "设备：{{ device_no_list }}，动作：[{{ service_name }}][{{ field_name }}]",
    
    "final_elem_output": (
        "{% if exec_info %}"
        "{% if iot_seq_output_cmd %}"
        "{% if iot_seq_cmd_as_guidance %}"
        "{{ cmd_info }}\n"
        "执行：\n"
        "{{ exec_info }}"
        "{% else %}"
        "{{ exec_info }}\n"
        "文案：\n"
        "{{ cmd_info }}"
        "{% endif %}"
        "{% else %}"
        "{{ exec_info }}"
        "{% endif %}"
        "{% else %}"
        "无"
        "{% endif %}"
    ),
}
