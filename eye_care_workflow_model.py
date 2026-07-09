"""
AI Eye Care Workflow Prototype
Author: Albert Du

What it does:
- Takes patient intake inputs
- Optionally accepts an OCT image
- Runs OCT classification if image is provided
- Optionally retrieves similar OCT cases using FAISS RAG
- Generates triage level
- Creates SOAP-style chart note
- Recommends follow-up scheduling
- Flags prior authorization / RCM needs
- Saves a dashboard-ready JSON summary

This is NOT a medical diagnostic model.
It is a workflow automation prototype for human review.
"""

import json
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Any


@dataclass
class PatientIntake:
    patient_name: str
    age: int
    symptoms: List[str]
    condition: str
    insurance: str
    visit_type: str
    last_injection_days_ago: int = 0
    has_imaging: bool = False
    transcript: str = ""
    image_path: Optional[str] = None


class EyeCareWorkflowModel:
    def __init__(self):
        self.urgent_symptoms = {
            "sudden vision loss",
            "severe eye pain",
            "flashes",
            "floaters",
            "curtain over vision",
            "trauma",
            "infection",
        }

        self.injection_conditions = {
            "wet amd",
            "diabetic macular edema",
            "dme",
            "cnv",
            "choroidal neovascularization",
            "retinal vein occlusion",
        }

        self.prior_auth_keywords = {
            "injection",
            "anti-vegf",
            "eylea",
            "lucentis",
            "vabysmo",
            "avastin",
            "surgery",
            "oct",
            "imaging",
        }

    def classify_oct_image(self, image_path: Optional[str]) -> Dict[str, Any]:
        if not image_path:
            return {
                "image_uploaded": False,
                "status": "No image provided",
            }

        image_file = Path(image_path)

        if not image_file.exists():
            return {
                "image_uploaded": True,
                "status": "Image path not found",
                "image_path": str(image_file),
            }

        try:
            from image_model.predict_image import predict_image

            prediction = predict_image(str(image_file))

            return {
                "image_uploaded": True,
                "status": "Classification completed",
                "image_path": str(image_file),
                "prediction": prediction,
            }

        except Exception as e:
            return {
                "image_uploaded": True,
                "status": "Classification failed",
                "image_path": str(image_file),
                "error": str(e),
            }

    def retrieve_similar_cases(self, image_path: Optional[str], top_k: int = 5) -> Dict[str, Any]:
        if not image_path:
            return {
                "retrieval_attempted": False,
                "status": "No image provided",
            }

        image_file = Path(image_path)

        if not image_file.exists():
            return {
                "retrieval_attempted": True,
                "status": "Image path not found",
                "image_path": str(image_file),
            }

        try:
            from image_model.RAG.rag_retrieve import retrieve_similar_images

            results = retrieve_similar_images(str(image_file), top_k=top_k)

            return {
                "retrieval_attempted": True,
                "status": "Similar case retrieval completed",
                "top_k": top_k,
                "similar_cases": results,
            }

        except Exception as e:
            return {
                "retrieval_attempted": True,
                "status": "Similar case retrieval not available yet",
                "error": str(e),
            }

    def triage_patient(self, intake: PatientIntake, oct_result: Dict[str, Any]) -> str:
        symptoms_lower = {s.lower() for s in intake.symptoms}
        condition_lower = intake.condition.lower()

        if symptoms_lower.intersection(self.urgent_symptoms):
            return "Urgent - same day physician review recommended"

        prediction_data = oct_result.get("prediction", {})
        predicted_class = ""

        if isinstance(prediction_data, dict):
            predicted_class = prediction_data.get("predicted_class", "").lower()

        if predicted_class in {"cnv", "dme"}:
            return "High priority - retina specialist review recommended"

        if intake.age >= 65 and condition_lower in self.injection_conditions:
            return "High priority - retina follow-up recommended"

        if predicted_class == "drusen":
            return "Moderate priority - monitor for AMD-related changes"

        if predicted_class == "normal":
            return "Routine - standard scheduling"

        return "Routine - standard scheduling"

    def recommend_schedule(self, intake: PatientIntake, oct_result: Dict[str, Any]) -> str:
        condition = intake.condition.lower()

        prediction_data = oct_result.get("prediction", {})
        predicted_class = ""

        if isinstance(prediction_data, dict):
            predicted_class = prediction_data.get("predicted_class", "").lower()

        if predicted_class in {"cnv", "dme"}:
            return "Schedule retina review promptly based on physician availability"

        if predicted_class == "drusen":
            return "Schedule non-urgent retina follow-up or monitoring visit"

        if condition in self.injection_conditions:
            if intake.last_injection_days_ago >= 42:
                return "Schedule injection follow-up as soon as possible"
            elif intake.last_injection_days_ago >= 28:
                return "Schedule within 1-2 weeks"
            else:
                return "Schedule next injection follow-up around 4-6 weeks from last injection"

        if "glaucoma" in condition:
            return "Schedule follow-up in 3-6 months depending on physician review"

        if "cataract" in condition:
            return "Schedule surgical evaluation or routine follow-up"

        return "Schedule routine eye care follow-up"

    def prior_auth_check(self, intake: PatientIntake, oct_result: Dict[str, Any]) -> Dict[str, str]:
        text = " ".join([
            intake.visit_type,
            intake.condition,
            " ".join(intake.symptoms),
            intake.transcript,
        ]).lower()

        prediction_data = oct_result.get("prediction", {})
        predicted_class = ""

        if isinstance(prediction_data, dict):
            predicted_class = prediction_data.get("predicted_class", "").lower()

        needs_auth = any(keyword in text for keyword in self.prior_auth_keywords)

        if intake.has_imaging or predicted_class in {"cnv", "dme", "drusen"}:
            needs_auth = True

        if needs_auth:
            return {
                "prior_auth_needed": "Yes",
                "reason": "Imaging, injection, treatment, or procedure may require authorization",
                "next_step": "Verify insurance benefits and prepare supporting documentation",
            }

        return {
            "prior_auth_needed": "No obvious flag",
            "reason": "No authorization-triggering workflow detected",
            "next_step": "Proceed with standard billing review",
        }

    def generate_chart_note(
        self,
        intake: PatientIntake,
        oct_result: Dict[str, Any],
        rag_result: Dict[str, Any],
    ) -> Dict[str, str]:
        symptoms = ", ".join(intake.symptoms)

        prediction_data = oct_result.get("prediction", {})
        predicted_class = "Not available"
        confidence = "Not available"

        if isinstance(prediction_data, dict):
            predicted_class = prediction_data.get("predicted_class", "Not available")
            confidence = prediction_data.get("confidence", "Not available")

        return {
            "Subjective": (
                f"{intake.patient_name}, age {intake.age}, presents for {intake.visit_type}. "
                f"Reported symptoms: {symptoms}."
            ),
            "Objective": (
                f"OCT image analyzed. Model prediction: {predicted_class}. "
                f"Confidence: {confidence}."
                if intake.has_imaging
                else "No imaging uploaded in this prototype."
            ),
            "Assessment": (
                f"Known or suspected condition: {intake.condition}. "
                f"AI-assisted OCT output is for workflow support only and requires physician review."
            ),
            "Plan": self.recommend_schedule(intake, oct_result),
            "RAG_Support": (
                rag_result.get("status", "Similar case retrieval not performed")
            ),
        }

    def dashboard_summary(self, intake: PatientIntake) -> Dict[str, Any]:
        oct_result = self.classify_oct_image(intake.image_path)
        rag_result = self.retrieve_similar_cases(intake.image_path, top_k=5)

        triage = self.triage_patient(intake, oct_result)
        schedule = self.recommend_schedule(intake, oct_result)
        auth = self.prior_auth_check(intake, oct_result)
        note = self.generate_chart_note(intake, oct_result, rag_result)

        return {
            "prototype_disclaimer": (
                "Research workflow prototype only. Not a medical diagnosis. "
                "All outputs require licensed physician review."
            ),
            "patient": intake.patient_name,
            "age": intake.age,
            "condition": intake.condition,
            "insurance": intake.insurance,
            "visit_type": intake.visit_type,
            "symptoms": intake.symptoms,
            "triage": triage,
            "recommended_schedule": schedule,
            "prior_auth": auth,
            "oct_classification": oct_result,
            "similar_case_retrieval": rag_result,
            "chart_note": note,
            "human_review_required": True,
            "recruiter_demo_summary": (
                "This prototype combines patient intake automation, OCT image classification, "
                "FAISS-based similar image retrieval, triage support, prior authorization flagging, "
                "and dashboard-ready JSON output."
            ),
        }

    def save_patient_record(self, summary: Dict[str, Any]) -> Path:
        data_dir = Path("patients")
        data_dir.mkdir(exist_ok=True)

        safe_name = summary["patient"].replace(" ", "_").replace("/", "_")
        filename = f"{safe_name}.json"

        output_path = data_dir / filename

        with open(output_path, "w") as f:
            json.dump(summary, f, indent=4)

        return output_path


def collect_intake_from_terminal() -> PatientIntake:
    name = input("Patient name: ")
    age = int(input("Age: "))

    symptoms_input = input("Symptoms, separated by commas: ")
    symptoms = [s.strip() for s in symptoms_input.split(",") if s.strip()]

    condition = input("Condition: ")
    insurance = input("Insurance: ")
    visit_type = input("Visit type: ")

    last_injection_days_ago = int(input("Days since last injection, use 0 if none: "))

    imaging_input = input("Is imaging available? yes/no: ").lower()
    has_imaging = imaging_input in ["yes", "y"]

    image_path = None
    if has_imaging:
        image_path = input("OCT image path, or press Enter to skip: ").strip()
        if image_path == "":
            image_path = None

    transcript = input("Visit transcript or notes: ")

    return PatientIntake(
        patient_name=name,
        age=age,
        symptoms=symptoms,
        condition=condition,
        insurance=insurance,
        visit_type=visit_type,
        last_injection_days_ago=last_injection_days_ago,
        has_imaging=has_imaging,
        transcript=transcript,
        image_path=image_path,
    )


def run_eye_care_workflow(intake: PatientIntake) -> Dict[str, Any]:
    model = EyeCareWorkflowModel()
    result = model.dashboard_summary(intake)
    saved_path = model.save_patient_record(result)

    result["saved_record_path"] = str(saved_path)

    return result


if __name__ == "__main__":
    intake = collect_intake_from_terminal()
    result = run_eye_care_workflow(intake)

    print(json.dumps(result, indent=2))