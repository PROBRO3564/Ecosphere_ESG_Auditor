#  EcoSphere: Carbon & Air Quality Analytics

EcoSphere is an AI-powered environmental auditing dashboard designed to track, calculate, and analyze carbon emissions and air quality impact (PM2.5) across transport sectors and daily consumption habits. 

## Key Features
* **Multi-Agent Collaborative Pipeline:** Built using the `google-genai` SDK, utilizing three specialized AI agents (NLP Extractor, Carbon Auditor, and Eco-Coach) working in sync to parse unstructured text logs, execute precise mathematical tracking, and deliver hyper-personalized coaching.
* **Deterministic Calculations:** Avoids LLM mathematical hallucinations by parsing unstructured natural language into structured data schemas via Pydantic, executing emission factors programmatically.
* **Carbon Offset Tracking:** Automatically rewards active green travel (walking, cycling) by calculating car-displacement carbon credits.

## Tech Stack
* **Frontend:** Streamlit
* **AI Model backend:** Gemini 2.5 Flash (`google-genai`)
* **Data Processing:** Pandas, NumPy, Pydantic