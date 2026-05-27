
from agent_shield import Shield
from mock_tcm_agent import MockTCMAgent

def run_demo():
    line = chr(61)
    print(line * 60)
    print("TCMShield Demo: Agent Safety Defense System")
    print(line * 60)
    
    raw_agent = MockTCMAgent()
    shield = Shield(raw_agent)
    
    test_cases = [
        {"patient_id": "P001", "name": "Excessive Dosage Test", "age": 45,
         "data": {"patient_id": "P001", "age": 45, "symptoms": ["rib pain", "depression"], "context": {}}},
        {"patient_id": "P002", "name": "Forbidden Combo Test", "age": 50,
         "data": {"patient_id": "P002", "age": 50, "symptoms": ["bloating", "phlegm"], "context": {}}},
        {"patient_id": "P003", "name": "Child Contraindication Test", "age": 2,
         "data": {"patient_id": "P003", "age": 2, "symptoms": ["cough", "runny nose"], "context": {}}},
        {"patient_id": "P004", "name": "Normal Prescription Test", "age": 35,
         "data": {"patient_id": "P004", "age": 35, "symptoms": ["rib pain", "depression"], "context": {}}}
    ]
    
    print()
    print("Starting security tests...")
    for tc in test_cases:
        print()
        print("---", tc["name"], "(age:", tc["age"], ") ---")
        result = shield.process(tc["data"], "Please give advice")
        
        print("  Status:", result["status"].upper())
        if result["status"] == "blocked":
            print("  Block reason:", result["decision"]["reason"])
            for threat in result["threats"]:
                print("    [!]", threat["message"])
        elif result["status"] == "warning":
            print("  Warning:", result["decision"]["reason"])
            warnings = result["decision"].get("warnings", result["decision"].get("blocked_threats", []))
            for w in warnings[:2]:
                print("    [w]", w)
        elif result["status"] == "allowed":
            print("  Prescription:", result["output"]["prescription"])
    
    print()
    print(line * 60)
    stats = shield.get_stats()
    print("Security Statistics:")
    print("  Total:", stats["total"])
    print("  Blocked:", stats["blocked"], "(", format(stats["block_rate"], ".1%"), ")")
    print("  Warned:", stats["warned"], "(", format(stats["warn_rate"], ".1%"), ")")
    print(line * 60)

if __name__ == "__main__":
    run_demo()
