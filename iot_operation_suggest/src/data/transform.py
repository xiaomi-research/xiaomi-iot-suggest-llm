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
IoT User Sequence Data Transformation Module

This module is responsible for converting raw user data into the format required for model training.
Main features:
1. Parse user historical operation intents
2. Generate prompt (pre_text) and label (post_text)
3. Support online_response field for DPO training
"""

import copy
import json
from datetime import datetime
from collections import OrderedDict
from typing import Dict, List, Any, Tuple, Optional

from jinja2 import Template

from .config import (
    IOT_USER_SEQ_SFT_PROMPT_CONFIG,
    WEEKDAY_MAP,
    USE_RAW_VALUE_ENV_ATTR,
    NO_VALUE_ACTION_ATTR,
)


def get_template_map() -> Dict[str, Template]:
    """
    Get template mapping, converting string templates in config to Jinja2 Template objects.
    
    Returns:
        Dict[str, Template]: Mapping from template name to Template object
    """
    template_map = {}
    for key, value in IOT_USER_SEQ_SFT_PROMPT_CONFIG.items():
        if key != "instruct":
            template_map[key] = Template(value, keep_trailing_newline=True)
    return template_map


def get_city_str(city_info: str, device_list: List[Dict]) -> str:
    """
    Get city string.
    
    Args:
        city_info: City information, format "Province,City" or single city name
        device_list: Device list, used as fallback for city information
        
    Returns:
        str: Formatted city string
    """
    if "," in city_info:
        province, city = city_info.split(",")
        city_str = province
        if city != "未知" and city != province:
            city_str += f"{city}"
    elif city_info:
        city_str = city_info
    else:
        city_str = "未知"

    if city_str == "未知":
        for device in device_list:
            if "country" in device and device["country"]:
                city_str = device["country"]
                break
    return city_str


def get_weather_str(weather_info: str) -> str:
    """
    Get weather string.
    
    Args:
        weather_info: Weather information as JSON string
        
    Returns:
        str: Formatted weather string
    """
    if not weather_info:
        return "未知"
    try:
        weather_data = json.loads(weather_info)
    except (json.JSONDecodeError, TypeError):
        return "未知"
    
    if not weather_data:
        return "未知"
    
    # Convert numeric types
    for field in ["temperature", "humidity", "pm25"]:
        if field in weather_data:
            try:
                weather_data[field] = float(weather_data[field])
            except (ValueError, TypeError):
                pass

    weather_fields = []
    
    if "weather" in weather_data and weather_data["weather"]:
        weather_fields.append(weather_data["weather"])
    
    if "temperature" in weather_data and isinstance(weather_data["temperature"], (int, float)):
        weather_fields.append(f"温度:{int(weather_data['temperature'])}")
    
    if "humidity" in weather_data and isinstance(weather_data["humidity"], (int, float)):
        weather_fields.append(f"湿度:{int(weather_data['humidity'])}")
    
    # Calculate air quality
    aqi = ""
    if "aqi" in weather_data and weather_data["aqi"]:
        aqi = weather_data["aqi"]
    elif "pm25" in weather_data and isinstance(weather_data["pm25"], (int, float)) and weather_data["pm25"] > 0:
        pm25 = weather_data["pm25"]
        if pm25 <= 20:
            aqi = "优"
        elif pm25 <= 35:
            aqi = "良"
        elif pm25 <= 75:
            aqi = "轻度污染"
        elif pm25 <= 115:
            aqi = "中度污染"
        elif pm25 <= 150:
            aqi = "重度污染"
        else:
            aqi = "严重污染"
    
    if aqi:
        weather_fields.append(f"空气质量:{aqi}")
    
    if weather_fields:
        return ", ".join(weather_fields)
    return "未知"


def get_device_info(
    device_list: List[Dict], 
    template_map: Dict[str, Template], 
    active_device_did: Optional[Dict[str, str]] = None
) -> Tuple[str, Dict[str, int]]:
    """
    Get available device information.
    
    Args:
        device_list: Device list
        template_map: Template mapping
        active_device_did: Mapping from active device DID to status
        
    Returns:
        Tuple[str, Dict[str, int]]: (Device info string, DID to number mapping)
    """
    if not active_device_did:
        active_device_did = {}
    
    id_by_did = {}
    device_str_list = []
    no = 1
    
    for device in device_list:
        members = device.get("members", [])
        room_name = device["room_name"] if device.get("room_name") else "未知房间"
        
        if len(members) > 1:
            for mem_idx, member in enumerate(members):
                new_did = device["did"] + f"#{mem_idx}"
                device_state = active_device_did.get(new_did, "未知")
                device_str = template_map["single_device_info"].render(
                    no=no, 
                    device_name=member["name"], 
                    category=device["category"],
                    room_name=room_name, 
                    device_state=device_state
                )
                device_str_list.append(device_str)
                id_by_did[new_did] = no
                no += 1
        else:
            device_state = active_device_did.get(device["did"], "未知")
            device_name = members[0]['name'] if members and device['device_name'] != members[0]['name'] else device['device_name']
            device_str = template_map["single_device_info"].render(
                no=no, 
                device_name=device_name, 
                category=device["category"],
                room_name=room_name, 
                device_state=device_state
            )
            device_str_list.append(device_str)
            id_by_did[device["did"]] = no
            no += 1
    
    device_info = "\n".join(device_str_list)
    return device_info, id_by_did


def get_single_intent_info(
    intent: Dict, 
    template_map: Dict[str, Template], 
    prev_env_info: Dict[str, float],
    merge_exec_info: bool = False
) -> Tuple[Dict[str, str], Dict[str, float]]:
    """
    Get information for a single intent.
    
    Args:
        intent: Intent data
        template_map: Template mapping
        prev_env_info: Previous environment information
        merge_exec_info: Whether to merge execution information
        
    Returns:
        Tuple[Dict[str, str], Dict[str, float]]: (Intent info dict, Current environment info)
    """
    logs = intent["logs"]
    queries = [q for q in intent.get("queries", []) if q] if intent.get("queries") else [""]
    action_time = intent["action_time"]
    _, day_type, week_day = intent["date_info"].split(",")
    week_day = WEEKDAY_MAP[week_day]
    
    elem_str_map = {}
    action_str_map = OrderedDict()
    cmd_info = "直接操控"
    feedback_info = ""
    crt_env_info = {}

    # Merge execution information
    if merge_exec_info:
        new_logs = []
        action_list_by_merged_key = OrderedDict()
        for log in logs:
            if log["elem_type"] != "action":
                new_logs.append(log)
            else:
                merged_key = f"{log['room_name']}#{log['service_name']}#{log['field_name']}"
                if log["field_name"] not in NO_VALUE_ACTION_ATTR:
                    merged_key += f"#{log['value']}"
                action_list_by_merged_key.setdefault(merged_key, []).append(log)
        
        for action_list in action_list_by_merged_key.values():
            log = copy.deepcopy(action_list[0])
            log["device_name"] = []
            for action in action_list:
                if action["device_name"] not in log["device_name"]:
                    log["device_name"].append(action["device_name"])
            new_logs.append(log)
        logs = new_logs

    for log in logs:
        elem_str = ""
        device_name = log.get("device_name", "未知")
        room_name = log.get("room_name", "未知房间") or "未知房间"
        
        if isinstance(device_name, list):
            device_name = [d.strip() for d in device_name]
        elif device_name:
            device_name = device_name.strip()
        room_name = room_name.strip()
        
        if log["elem_type"] == "env":
            env_key = f"{room_name}#{log['field_name']}"
            env_value = float(log["value"])
            crt_env_info[env_key] = env_value
            
            if env_key in prev_env_info:
                prev_env_value = prev_env_info[env_key]
                if abs(prev_env_value - env_value) / (prev_env_value + 0.000001) < 0.01:
                    continue
            
            value = log.get("value_norm", log["value"])
            if log["field_name"] in USE_RAW_VALUE_ENV_ATTR:
                value = round(float(log["value"]))
            elem_str = template_map["single_env_info"].render(
                room_name=room_name, 
                attribute=log["field_name"], 
                value=value
            )
            
        elif log["elem_type"] == "event":
            field_name = log["field_name"]
            if field_name == "门锁事件":
                field_name = log["value"]
            if "piid" in field_name or "value" in field_name:
                continue
            
            env_key = f"{room_name}#{field_name}"
            crt_env_info[env_key] = 1
            if env_key in prev_env_info:
                continue
            
            elem_str = template_map["single_event_info"].render(
                room_name=room_name, 
                attribute=field_name
            )
            
        elif log["elem_type"] == "action":
            nice_client = log.get("nice_client", "")
            
            if nice_client == "miioproc-openhome":
                cmd_info = "语音操控"

            elem_str = template_map["single_exec_info"].render(
                device_name=device_name, 
                room_name=room_name,
                service_name=log["service_name"], 
                field_name=log["field_name"]
            )
            if log["field_name"] not in NO_VALUE_ACTION_ATTR:
                elem_str += f"[{log['value']}]"
            action_str_map.setdefault(log.get("category", ""), []).append(elem_str)
        
        if elem_str and log["elem_type"] != "action":
            elem_str_map.setdefault(log["elem_type"], []).append(elem_str)
    
    if cmd_info == "语音操控" and queries:
        dedup_queries = []
        for query in queries:
            if query.startswith("psk"):
                continue
            if query not in dedup_queries:
                dedup_queries.append(query)
        query_str = "、".join(dedup_queries)
        if query_str:
            cmd_info += f"-{query_str}"

    action_time_obj = datetime.fromtimestamp(action_time)
    time_info = f"{action_time_obj.strftime('%Y-%m-%d %H:%M:%S')}，{week_day}，{day_type}"
    
    env_info = "无"
    if "env" in elem_str_map:
        env_info = "\n".join(elem_str_map['env'])
    
    event_info = "无"
    if "event" in elem_str_map:
        event_info = "\n".join(elem_str_map['event'])
    
    exec_info = "\n".join([e for el in action_str_map.values() for e in el])

    
    single_intent_info = OrderedDict(
        time_info=time_info,
        env_info=env_info,
        event_info=event_info,
        cmd_info=cmd_info,
        exec_info=exec_info,
        feedback_info=feedback_info,
    )
    
    return single_intent_info, crt_env_info


def get_truncated_sequence(
    intents: List[Dict], 
    hist_limit_value: int = 50
) -> List[Dict]:
    """
    Get truncated intent sequence.
    
    Args:
        intents: Intent sequence
        hist_limit_value: History record limit count
        
    Returns:
        List[Dict]: Truncated intent sequence
    """
    if len(intents) == 1:
        return intents
    
    hist_intents = intents[:-1]
    
    if hist_limit_value > 0:
        new_hist_intents = hist_intents[-hist_limit_value:]
    elif hist_limit_value == 0:
        new_hist_intents = []
    else:
        new_hist_intents = hist_intents
    
    return new_hist_intents + intents[-1:]


def iot_user_seq_sft_single_transform(
    intents: List[Dict], 
    device_list: List[Dict], 
    template_map: Dict[str, Template], 
    city_str: str, 
    weather_str: str = "",
    merge_exec_info: bool = True, 
    iot_seq_output_cmd: bool = False,
    iot_seq_cmd_as_guidance: bool = False
) -> Tuple[str, str]:
    """
    Perform data transformation on a single sequence.
    
    Args:
        intents: Intent sequence
        device_list: Device list
        template_map: Template mapping
        city_str: City string
        weather_str: Weather string
        merge_exec_info: Whether to merge execution information
        iot_seq_output_cmd: Whether to output command
        iot_seq_cmd_as_guidance: Whether command is used as guidance
        
    Returns:
        Tuple[str, str]: (pre_text, post_text)
    """
    history_intent_str_list = []
    final_intent_trigger_str = ""
    final_intent_action_str = ""

    prev_env_info = {}
    for idx, intent in enumerate(intents):
        is_final_intent = idx == len(intents) - 1
        logs = intent["logs"]

        single_intent_info, prev_env_info = get_single_intent_info(
            intent, template_map, prev_env_info, merge_exec_info
        )

        if is_final_intent:
            # Process active device information
            active_device_info = {}
            active_service_by_did = {}
            for elem in logs:
                if elem["elem_type"] == "device":
                    if elem.get("service_name", "") not in active_service_by_did.get(elem["did"], set()):
                        active_device_info.setdefault(elem["did"], []).append(elem)
                        active_service_by_did.setdefault(elem["did"], set()).add(elem.get("service_name", ""))
            
            active_device_did = {}
            for did, elems in active_device_info.items():
                if len(elems) == 1:
                    active_device_did[did] = elems[0]["value"]
                else:
                    active_device_did[did] = "、".join([
                        f"{elem.get('service_name', '')}-{elem['value']}" for elem in elems
                    ])

            device_info, id_by_did = get_device_info(device_list, template_map, active_device_did)

            # Merge action information
            merge_action_map = {}
            for log in logs:
                if log["elem_type"] == "action" and log["did"] in id_by_did:
                    merge_action_map.setdefault(
                        (log.get("service_name", ""), log.get("field_name", ""), log.get("value", "")), 
                        []
                    ).append(id_by_did[log["did"]])
            
            exec_str_list = []
            for (service_name, field_name, value), id_list in merge_action_map.items():
                id_list.sort()
                device_no_list = ",".join([str(i) for i in id_list])
                exec_str = template_map["merged_exec_info"].render(
                    device_no_list=device_no_list, 
                    service_name=service_name, 
                    field_name=field_name
                )
                if field_name not in NO_VALUE_ACTION_ATTR:
                    exec_str += f"[{value}]"
                exec_str_list.append(exec_str)
            
            if exec_str_list:
                exec_info = "\n".join(exec_str_list)
                single_intent_info["iot_seq_output_cmd"] = iot_seq_output_cmd
            else:
                exec_info = "无"
                single_intent_info["iot_seq_output_cmd"] = False

            single_intent_info["iot_seq_cmd_as_guidance"] = iot_seq_cmd_as_guidance
            single_intent_info["exec_info"] = exec_info
            single_intent_info["device_list"] = device_info
            single_intent_info["cmd_info"] = intent.get("queries", [""])[0] if intent.get("queries") else ""

        if is_final_intent:
            single_intent_info["city_info"] = city_str
            single_intent_info["weather_info"] = weather_str
            final_intent_trigger_str = template_map["final_elem_input"].render(**single_intent_info)
            final_intent_action_str = template_map["final_elem_output"].render(**single_intent_info)
        else:
            history_intent_str = template_map["single_elem"].render(**single_intent_info)
            history_intent_str_list.append(history_intent_str)

    pre_text = IOT_USER_SEQ_SFT_PROMPT_CONFIG["instruct"] + template_map["template"].render(
        history="\n\n".join(history_intent_str_list), 
        final_intent=final_intent_trigger_str
    )
    post_text = final_intent_action_str

    return pre_text, post_text


def iot_user_seq_sft_transform(
    sample: Dict[str, Any], 
    template_version: str = "v1_weather",
    iot_seq_output_cmd: bool = False,
    eval_hist_limit_value: int = 50,
    merge_exec_info: bool = True,
    iot_seq_cmd_as_guidance: bool = False
) -> Dict[str, str]:
    """
    IoT user sequence SFT data transformation main function.
    
    Converts raw samples into the format required for model training:
    - pre_text: Used as prompt input
    - post_text: Used as label/response
    - online_response: Online model response (for DPO training)
    
    Args:
        sample: Raw sample data containing the following fields:
            - device_list: Device list
            - history_intents: Historical intent sequence
            - actual_intents: Current intents
            - city_info: City information
            - weather_info: Weather information
            - online_response: Online model response
        template_version: Template version, currently only supports v1_weather
        iot_seq_output_cmd: Whether to output command
        eval_hist_limit_value: History record limit count
        merge_exec_info: Whether to merge execution information
        iot_seq_cmd_as_guidance: Whether command is used as guidance
        
    Returns:
        Dict[str, str]: Contains the following fields:
            - pre_text: Prompt text
            - post_text: Response text
            - online_response: Online model response
    """
    template_map = get_template_map()

    device_list = sample["device_list"]
    history_intents = sample["history_intents"]
    actual_intents = sample["actual_intents"]

    assert actual_intents, "actual intents must be non-empty."

    city_str = get_city_str(sample.get("city_info", ""), device_list)
    weather_str = get_weather_str(sample.get("weather_info", ""))

    all_intents = history_intents + actual_intents
    sequence = get_truncated_sequence(all_intents, eval_hist_limit_value)

    pre_text, post_text = iot_user_seq_sft_single_transform(
        sequence, 
        device_list, 
        template_map, 
        city_str, 
        weather_str,
        merge_exec_info=merge_exec_info, 
        iot_seq_output_cmd=iot_seq_output_cmd,
        iot_seq_cmd_as_guidance=iot_seq_cmd_as_guidance
    )

    online_response = sample.get("online_response", "")

    return {
        "pre_text": pre_text,
        "post_text": post_text,
        "online_response": online_response
    }
