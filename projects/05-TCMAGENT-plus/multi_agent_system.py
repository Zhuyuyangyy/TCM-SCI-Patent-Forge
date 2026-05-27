# TCM Multi-Agent Consultation System
from tongue_agent import TongueAgent
from history_agent import HistoryAgent
from lab_agent import LabAgent
from deliberation import DeliberationModule
from typing import Dict

class TCMConsultationSystem:
    def __init__(self):
        self.tongue_agent = TongueAgent()
        self.history_agent = HistoryAgent()
        self.lab_agent = LabAgent()
        self.deliberation = DeliberationModule()
    
    def consult(self, patient_data: Dict) -> Dict:
        sep = "=" * 50
        print(sep)
        print("TCMAgent++ Multi-Agent Consultation System")
        print(sep)
        
        print()
        print("[Step 1] Parallel Analysis by Expert Agents...")
        
        tongue_evidence = self.tongue_agent.process(patient_data)
        print(f"  TongueAgent: {tongue_evidence.get('syndrome', 'N/A')}")
        
        history_evidence = self.history_agent.process(patient_data)
        print(f"  HistoryAgent: {history_evidence.get('primary_syndrome', 'N/A')}")
        
        lab_evidence = self.lab_agent.process(patient_data)
        print(f"  LabAgent: {lab_evidence.get('tcm_correlation', {}).get('primary_correlation', 'N/A')}")
        
        print()
        print("[Step 2] Evidence Collection...")
        all_evidence = [tongue_evidence, history_evidence, lab_evidence]
        
        print()
        print("[Step 3] Collaborative Deliberation...")
        result = self.deliberation.deliberate(all_evidence)
        
        print(f"  Final Syndrome: {result['final_syndrome']}")
        print(f"  Confidence: {result['confidence']:.2%}")
        
        if result['conflicts']:
            print(f"  Detected {len(result['conflicts'])} conflicts, resolved")
        
        return {
            'patient_id': patient_data.get('patient_id', 'unknown'),
            'final_syndrome': result['final_syndrome'],
            'treatment_principle': result['treatment_principle'],
            'confidence': result['confidence'],
            'syndrome_votes': result['syndrome_votes'],
            'reasoning': result['reasoning'],
            'evidence': all_evidence
        }
