#!/usr/bin/env python3
"""D01 Demo - Damp-Heat Female -> Digital Twin -> Intervention -> Risk Report"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import constitution_state as cs
import transition_model as tm
import counterfactual_engine as ce
import visualization as viz


def print_header(title):
    print("\n" + "=" * 70)
    print("  " + title)
    print("=" * 70)


def print_section(title):
    print("\n[" + title + "]")


def main():
    print_header("D01 Constitution Digital Twin - Damp-Heat Female Intervention Demo")

    # Step 1: Create Digital Twin
    print_section("Step 1: Create Damp-Heat Female Digital Twin")
    baseline = cs.ConstitutionStateBuilder.create_damp_heat_female(patient_id="DHF2026001", age=35)

    print("Patient ID: " + baseline.patient_id)
    print("Age: " + str(baseline.age))
    print("Gender: " + baseline.gender)
    dom = baseline.get_dominant_constitution()
    print("Dominant: " + dom[0] + " (" + str(dom[1])[:4] + ")")
    print("Yin-Yang: " + baseline.get_yin_yang_state().value)
    print("Qi-Blood: " + baseline.get_qi_blood_state().value)

    print("\nConstitution Scores:")
    for c, s in sorted(baseline.constitution_scores.items(), key=lambda x: -x[1]):
        bar = "=" * int(s * 20)
        print("  " + c + ": " + str(s)[:4] + " " + bar)

    # Step 2: Setup Interventions
    print_section("Step 2: Setup Intervention Scenarios")
    engine = ce.CounterfactualEngine(baseline)

    scenarios = {
        "Plan A - Exercise": [("运动", 180)],
        "Plan B - Exercise+Diet": [("运动", 180), ("饮食调节", 180)],
        "Plan C - Combined": [("运动", 180), ("饮食调节", 180), ("推拿", 60), ("艾灸", 60)],
        "Plan D - Enhanced": [("运动", 180), ("饮食调节", 180), ("推拿", 90), ("艾灸", 90)]
    }

    for name, interventions in scenarios.items():
        engine.create_scenario(name, interventions)
        print("  + " + name + " created")
        for int_type, days in interventions:
            print("      - " + int_type + ": " + str(days) + " days")

    # Step 3: Run Simulation
    print_section("Step 3: Run Counterfactual Simulation")
    results = engine.compare_scenarios()

    balanced_key = "平和质"

    for name, result in results.items():
        dom_change = result["dominant_change"]
        init_bal = result["initial_state"].constitution_scores.get(balanced_key, 0.0)
        final_bal = result["final_state"].constitution_scores.get(balanced_key, 0.0)
        print("  [" + name + "]")
        print("    Dominant: " + dom_change[0] + " -> " + dom_change[1])
        print("    Balanced: " + str(init_bal)[:4] + " -> " + str(final_bal)[:4])
        print("    Score: " + str(result["improvement_score"])[:5])

    # Step 4: Risk Report
    print_section("Step 4: Disease Risk Assessment")
    init_risks = ce.DISEASE_RISKS.get(baseline.get_dominant_constitution()[0], {})
    print("Initial Risks:")
    for r, s in sorted(init_risks.items(), key=lambda x: -x[1]):
        bar = "=" * int(s * 20)
        print("  " + r + ": " + str(s)[:4] + " " + bar)

    best_name = max(results.keys(), key=lambda k: results[k]["improvement_score"])
    best_result = results[best_name]
    final_risks = best_result["final_risks"]

    print("\n" + best_name + " Risk Changes:")
    print("  " + "{:15}".format("") + " {:>8}".format("Before") + " {:>8}".format("After") + " {:>8}".format("Change"))
    print("  " + "-" * 40)
    all_risks = set(init_risks.keys()) | set(final_risks.keys())
    for risk in sorted(all_risks):
        init = init_risks.get(risk, 0)
        final = final_risks.get(risk, 0)
        change = final - init
        arrow = "DOWN " if change < -0.01 else ("UP   " if change > 0.01 else "SAME ")
        print("  " + "{:15}".format(risk) + " {:8.2f}".format(init) + " {:8.2f}".format(final) + " " + arrow + "{:.2f}".format(abs(change)))

    # Step 5: Visualizations (text-based)
    print_section("Step 5: Text-Based Visualizations")
    output_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        viz.DashboardGenerator.generate(baseline, [], {name: results[name]["improvement_score"] for name in results}, list(init_risks.values()))
    except Exception as e:
        print("  Visualization: " + str(e))

    # Step 6: Text Report
    print_section("Step 6: Generate Text Report")
    report = engine.generate_report()
    print(report)

    report_path = os.path.join(output_dir, "intervention_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print("\nReport saved: " + report_path)

    # Summary
    init_bal_final = results[best_name]["initial_state"].constitution_scores.get(balanced_key, 0.0)
    final_bal_final = results[best_name]["final_state"].constitution_scores.get(balanced_key, 0.0)

    print_header("Demo Complete")
    print("Summary:")
    print("1. Baseline: Damp-Heat Female (" + baseline.patient_id + ")")
    print("2. Best Plan: " + best_name)
    print("3. Dominant Change: " + best_result["dominant_change"][0] + " -> " + best_result["dominant_change"][1])
    print("4. Balanced Gain: " + str(init_bal_final)[:4] + " -> " + str(final_bal_final)[:4])
    print("5. Improvement Score: " + str(best_result["improvement_score"])[:5])
    print("\nOutput Files:")
    print("  - intervention_report.txt")
    print("  - Text visualizations (printed above)")


if __name__ == "__main__":
    main()
