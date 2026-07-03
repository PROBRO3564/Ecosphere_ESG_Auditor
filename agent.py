import os
import json
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# ==========================================
# 1. DATA STRUCTURES & SCHEMAS
# ==========================================
class ExtractedEntity(BaseModel):
    raw_name: str = Field(description="The exact name used by the user, e.g., 'Thar', 'R15', 'bus'")
    resolved_type: str = Field(description="Must be one of: 'flight', 'bus', 'train', 'motorbike', 'car', 'auto-rickshaw', 'bottle', 'can', 'cup', 'cycling', 'walking', 'running'")
    value: float = Field(description="The numeric distance or quantity extracted")
    unit: str = Field(description="'km' for transit/active travel, 'items' for consumables")

class NormalizationPayload(BaseModel):
    entities: List[ExtractedEntity]

class ActivityItem(BaseModel):
    category: str
    metric: str
    value: float
    unit: str
    co2_kg: float
    pm25_g: float

class CarbonFootprintReport(BaseModel):
    agent_reasoning_logs: List[str] = Field(description="Internal step-by-step reasoning logs from the agent team execution")
    activities: List[ActivityItem]
    total_net_co2_kg: float
    total_pm25_g: float
    green_actionable_tip: str

# ==========================================
# 2. INITIALIZE CLIENT
# ==========================================
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
# CONSTANTS FOR AUDITOR CALCULATIONS
EMISSION_FACTORS = {
    "flight": {"co2": 2.50, "pm25": 0.015, "cat": "Public & Shared Transit", "lbl": "Commercial Flight"},
    "bus": {"co2": 0.12, "pm25": 0.035, "cat": "Public & Shared Transit", "lbl": "Public Transit Bus"},
    "train": {"co2": 0.04, "pm25": 0.000, "cat": "Public & Shared Transit", "lbl": "Passenger Rail Train"},
    "motorbike": {"co2": 0.05, "pm25": 0.012, "cat": "Personal Transport", "lbl": "Bike/Motorbike"},
    "car": {"co2": 0.20, "pm25": 0.030, "cat": "Personal Transport", "lbl": "Car/Drive"},
    "auto-rickshaw": {"co2": 0.07, "pm25": 0.010, "cat": "Public & Shared Transit", "lbl": "Auto-rickshaw"},
    "bottle": {"co2": 0.25, "pm25": 0.000, "cat": "Lifestyle & Consumption", "lbl": "Plastic Water Bottle"},
    "can": {"co2": 0.18, "pm25": 0.000, "cat": "Lifestyle & Consumption", "lbl": "Cold Drink Can"},
    "cup": {"co2": 0.05, "pm25": 0.000, "cat": "Lifestyle & Consumption", "lbl": "Paper Cup"},
    "cycling": {"co2": 0.00, "pm25": 0.000, "cat": "Active Green Transit", "lbl": "Bicycle (Human Powered)"},
    "walking": {"co2": 0.00, "pm25": 0.000, "cat": "Active Green Transit", "lbl": "Walking / Foot Travel"},
    "running": {"co2": 0.00, "pm25": 0.000, "cat": "Active Green Transit", "lbl": "Running / Jogging"},
}

# ==========================================
# 3. AGENT PIPELINE EXECUTION
# ==========================================
def run_agent_pipeline(user_input: str) -> CarbonFootprintReport:
    logs = []
    
    # --------------------------------------
    # AGENT 1: The Extraction & NLP Specialist
    # --------------------------------------
    logs.append("[Agent 1 - NLP Extractor]: Spawning instance. Analyzing unstructured log tokens...")
    
    prompt_agent_1 = f"""
    Analyze this text log and extract all numerical quantities linked to transit or lifestyle item metrics.
    Map custom names to their standard base categories (e.g., 'Thar' -> 'car', 'R15' -> 'motorbike', 'auto' -> 'auto-rickshaw').
    
    User Log: "{user_input}"
    """
    
    response_1 = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt_agent_1,
        config=types.GenerateContentConfig(
            system_instruction="You are an expert NLP Entity Extractor. Isolate quantities and resolve them to standard labels strictly using the schema.",
            response_mime_type="application/json",
            response_schema=NormalizationPayload,
            temperature=0.0
        )
    )
    
    parsed_payload = NormalizationPayload.model_validate_json(response_1.text)
    logs.append(f"[Agent 1 - NLP Extractor]: Extraction complete. Identified {len(parsed_payload.entities)} active environmental nodes.")
    for ent in parsed_payload.entities:
        logs.append(f"   -> Found Entity: '{ent.raw_name}' mapped to type '{ent.resolved_type}' with value {ent.value}")

    # --------------------------------------
    # AGENT 2: The Carbon Auditor (Math Execution)
    # --------------------------------------
    logs.append("[Agent 2 - Carbon Auditor]: Loading verified emission coefficients. Commencing execution...")
    processed_activities = []
    total_active_km = 0.0
    
    for item in parsed_payload.entities:
        rtype = item.resolved_type
        if rtype in EMISSION_FACTORS:
            factor = EMISSION_FACTORS[rtype]
            
            # Perform programmatic mathematical execution (No LLM math hallucinations)
            co2_calc = round(item.value * factor["co2"], 2)
            pm_calc = round(item.value * factor["pm25"], 3)
            
            display_label = f"{factor['lbl']} ({item.raw_name})" if item.raw_name.lower() != rtype else factor["lbl"]
            
            processed_activities.append(ActivityItem(
                category=factor["cat"],
                metric=display_label,
                value=item.value,
                unit=item.unit,
                co2_kg=co2_calc,
                pm25_g=pm_calc
            ))
            
            if factor["cat"] == "Active Green Transit":
                total_active_km += item.value
                
    # Calculate offset credit programmatically
    if total_active_km > 0:
        saved_co2 = round(total_active_km * 0.20, 2)
        saved_pm = round(total_active_km * 0.030, 3)
        logs.append(f"[Agent 2 - Carbon Auditor]: Calculating green offsets for {total_active_km} km of active transit.")
        processed_activities.append(ActivityItem(
            category="Environmental Offsets",
            metric="Car Displacement Savings (Active Commute)",
            value=total_active_km,
            unit="km",
            co2_kg=-saved_co2,
            pm25_g=-saved_pm
        ))

    net_co2 = round(sum(act.co2_kg for act in processed_activities), 2)
    net_pm = round(sum(act.pm25_g for act in processed_activities), 3)
    logs.append(f"[Agent 2 - Carbon Auditor]: Audit finalized. Net Carbon Balance: {net_co2} kg, PM2.5: {net_pm} g.")

    # --------------------------------------
    # AGENT 3: The Eco-Coach (Strategic Advisory)
    # --------------------------------------
    logs.append("[Agent 3 - Eco-Coach]: Reviewing full carbon audit ledger to synthesize actionable behavioral feedback...")
    
    summary_for_coach = {
        "net_co2_kg": net_co2,
        "activities": [{"metric": a.metric, "co2": a.co2_kg} for a in processed_activities if a.co2_kg > 0]
    }
    
    prompt_agent_3 = f"Analyze this footprint data sheet and write one punchy, hyper-personalized green tip addressing the user's single largest emission sector:\n{json.dumps(summary_for_coach)}"
    
    response_3 = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt_agent_3,
        config=types.GenerateContentConfig(
            system_instruction="You are an elite sustainability advisor. Review the ledger and output a single, direct, realistic advice sentence. Do not mention JSON structures.",
            temperature=0.3
        )
    )
    
    actionable_tip = response_3.text.strip()
    logs.append("[Agent 3 - Eco-Coach]: Strategy generated successfully.")

    # Assemble the final unified report state
    return CarbonFootprintReport(
        agent_reasoning_logs=logs,
        activities=processed_activities,
        total_net_co2_kg=net_co2,
        total_pm25_g=net_pm,
        green_actionable_tip=actionable_tip
    )

def analyze_expenses(user_input: str) -> CarbonFootprintReport:
    """Streamlit interface wrapper entrypoint"""
    return run_agent_pipeline(user_input)
