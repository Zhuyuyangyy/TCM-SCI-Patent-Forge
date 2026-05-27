# TCMAgent++ Multi-Agent TCM Consultation System Demo
from multi_agent_system import TCMConsultationSystem

def run_demo():
    print("TCMAgent++ Multi-Agent TCM Consultation System Demo")
    print("=" * 60)
    
    patient_data = {
        "patient_id": "P001",
        "tongue": "舌红，苔黄腻",
        "symptoms": ["胁肋胀痛", "情绪抑郁", "食欲不振", "失眠多梦", "口苦"],
        "duration": "3个月",
        "history": "慢性乙肝病史",
        "lab_results": {
            "ALT": 85.0,
            "AST": 72.0,
            "Cr": 95.0
        }
    }
    
    system = TCMConsultationSystem()
    report = system.consult(patient_data)
    
    print()
    print("=" * 60)
    print("Consultation Report")
    print("=" * 60)
    print(f"Patient ID: {report['patient_id']}")
    print(f"Final Syndrome: {report['final_syndrome']}")
    print(f"Treatment Principle: {report['treatment_principle']}")
    print(f"Confidence: {report['confidence']:.2%}")
    print(f"Reasoning: {report['reasoning']}")
    print(f"Syndrome Votes: {report['syndrome_votes']}")

if __name__ == "__main__":
    run_demo()
