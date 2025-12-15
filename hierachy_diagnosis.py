from agents import Agent
from utils import extract_option, count_token_usage
# Remeber to replace words in <> in prompts.

def level_2_diagnosis(dataset, medical_case, medical_image, level1_report, token_usage):
    print("\n=====[LEVEL-2] Extra Expert Assessment=====")
    expert = Agent()

    query = """[Core Identity] You are an authoritative senior <MEDICAL FIELD> expert, highly proficient in <IMAGING MODALITIES> interpretation and diagnostic reasoning. Your role is to critically verify the consensus diagnosis made by two prior <MEDICAL FIELD> experts, ensuring it is logically sound, evidence-based, and consistent with <IMAGING MODALITIES> image features. [Task Focus] 1. First check the input image and read the question. 2. Evaluate whether the shared judgment aligns with the observed image findings and <IMAGING MODALITIES> criteria. 3. Identify any potential misinterpretation or overconfidence. 4. If their consensus is valid, reaffirm it; if not, provide your corrected final diagnosis. [Current Case] {medical_case}. [Previous Reports] {level1_report}. [Output Format] #Review Reasoning: <Write a rigorous 3-5 sentence paragraph explaining (1) the observed image evidence, (2) the logic of the prior judgments, (3) potential flaws or confirmations, (4) your diagnostic reasoning, and (5) your conclusion.> #Answer: <a single letter of your choice, e.g. A or B>.""".format(medical_case=medical_case,level1_report=level1_report)

    response = expert.chat(query, medical_image, 0.5)

    try:
        level2_option = extract_option(response.split('#')[-1].split(':')[-1])
    except:
        level2_option = extract_option(response.split(':')[-1])

    print(f'[LEVEL-2][Agent-extra:{expert.model_info}][Assessment]\n',response)

    token_usage = count_token_usage(token_usage, expert.get_token_usage())

    return level2_option, response, token_usage

def level_3_diagnosis(medical_case, medical_image, latest_report, level1_report_dict, token_usage):
    print("\n=====[LEVEL-3] Expert Panel Debate=====")
    print(f"{len(level1_report_dict)} experts are recruited to check options: {[o for o in level1_report_dict.keys()]}")

    panel_leader = Agent()
    debate_agents = {}
    init_critics = []
    
    for i, option in enumerate(level1_report_dict.keys()):
        debate_agent = Agent()
        query = """[Core Identity] You are an expert Critical Analyst, functioning as a Hypothesis Auditor. First check the input image, and read the question. Your task is to provide a balanced, objective, and rigorous review of a proposed hypothesis based on the provided source evidence. Your goal is to assess the overall viability and logical soundness of the hypothesis, not to attack it. You are assigned to uncover potential risks in option {OPTION} in the medical case and the supportive statements of option {OPTION} in [Historical Reports]. You should raise the risk that "why this hypothesis may be wrong", and your report would be given to a leader to make a decision. [Medical Case] {medical_case}. [Historical Reports] {latest_report}. [Output Format]#Flaws: <Describe the specific logical flaw, risk, or overlooked possibility in 3-5 CONCISE sentences.> Counter Evidence: <Cite specific evidence from the original case supporting your critique in 4 sentences.>.""".format(OPTION=option,medical_case=medical_case,latest_report=latest_report)
        critic = debate_agent.chat(query, medical_image, 0.5)

        print(f'\n[LEVEL-3][Agent{i+1}:{debate_agent.model_info}][Critics on ({option.upper()})]\n', critic)
        init_critics.append((option,critic))
        debate_agents[f"{option}"] = debate_agent

    init_critics = "[LEVEL-3 Expert Panel Critics]\n" + "\n".join([f"Critic Expert {i+1}: {critic[1]}" for i,critic in enumerate(init_critics)])

    leader_query = """[Core Identity] You are the Lead Adjudicator, responsible for chairing an expert critical analysis of conflicting hypotheses. You are impartial, perceptive, and skilled at uncovering the truth through precise inquiry. [Task 1] First check the input image and read the question. You have just received the initial arguments on a medical case from the Critic Specialists. Your task is not to form your own opinion yet, but to act as a rigorous, impartial critic. You must critically analyze each review below, identify its single biggest weakness, logical flaw, or unsupported assumption, and formulate a targeted, challenging question for each specialist, the question should help you solve the case. [Inquiry Methodology] Strictly follow these steps in your thinking: 1.Synthesize Critiques: Comprehensively read and understand the report submitted by each Hypothesis Auditor. 2.Identify Core Conflict: What is the central point of disagreement or the most critical identified risk among the competing audits? 3.Formulate Targeted Questions: Based on this core conflict, design a challenging question for each auditor that forces them to defend their critique. [Output Format] Inquiries:@ To Expert <Expert No., e.g 1> who reviews <The option it reviews, e.g A>: <The single, most pointed question for the Expert who reviews Option, based on the risks they identified in their report.> @ To Expert <Expert No., e.g 2> who reviews <The option it reviews, e.g B>: <The single, most pointed question for the Expert who reviews Option>...(until each expert in [Critics on Assessments] is inquired, no other contents). [Medical Case] {medical_case}. [Initial Independent Assessments] {latest_report}. [Critics on Assessments] {risek_report}. Now, begin your inquiry and output strictly according to the format and requirements:""".format(medical_case=medical_case,latest_report=latest_report, risk_report=init_critics)
    leader_consulations = panel_leader.chat(leader_query, medical_image, 0.1)
    print('\n[LEVEL-3][Leader Inquiries]\n', leader_consulations)

    consulations = leader_consulations.strip().split('@')[1:]
    consulations = [(extract_option(consulation.split(":")[0].split(' ')[-1]), consulation.split(":")[1].strip()) for consulation in consulations if len(consulation)>1]
    
    rebuttals = []
    for consul in consulations:
        option, inquiry = consul
        if option not in debate_agents.keys():
            continue
        agent = debate_agents[option]
        rebuttal_query ="""Please answer the question from the leader toward your support report in 1-3 sentences, do not change your stance:{inquiry}.""".format(inquiry=inquiry)
        rebuttal = agent.chat(rebuttal_query, medical_image, 0.1)
        print(f'\n[LEVEL-3][Critic for {option} - response]\n', rebuttal)
        rebuttals.append(f'[Critic for {option} - response]\n{rebuttal}')
        token_usage = count_token_usage(token_usage, agent.get_token_usage())

    rebuttals = "[Expert Panel Response]\n" + '\n'.join(rebuttals)
    level3_reports = init_critics + '[Leader Inquiries]' + leader_consulations + rebuttals

    leader_report_query = """[Response to your inquiries] {rebuttals} [Task 2] You have received all critiques and the final responses to your inquiries. Your task is to render the final, binding verdict on this case. Your decision must be based on which hypothesis best survived the logical stress test. [Adjudication Methodology] Strictly follow these steps in your thinking: 1. Global Review: Re-examine the complete record: the source evidence, the Critique Reports from each Critic Agent, your inquiries, and the Critics' final responses to those inquiries. 2. Compare Critique Impact: Your primary task is to compare the severity and impact of the flaws identified. Synthesize all information to determine which hypothesis, after rigorous scrutiny, best survived its dedicated critique. 3. Justify the Verdict: You must explicitly state why one hypothesis survived better than the other(s). Your final reasoning MUST be based on this direct comparison. 4. Render Final Verdict: Formulate your final, reasoned judgment, you can choose an overlooked choice when you are very confident after careful thinking. [Strict Instruction] This is the final step. No further escalation is possible. [Strict Output Format] #Final Reasoning: <A report, within 6-8 sentences, summarizing the comparative impact of the critiques. This must explain the rationale for your final verdict.> #Final Answer: <Only the single letter of your choice, e.g., A or B>.""".format(rebuttals=rebuttals)

    final_leader = Agent()
    leader_final_report = final_leader.chat(leader_report_query, medical_image, 0.1)
        
    level3_option = leader_final_report.split(':')[-1].strip()
    if 'None' in leader_final_report:
        level3_option = [item for item in level1_report_dict.keys()][0]

    print(f'\n[LEVEL-3][Panel Leader:{panel_leader.model_info}][Final Report]', level3_option)
    print(leader_final_report)

    level3_report = '[LEVEL-3 Expert Panel Discussions]\n' + init_critics + '[Panel Leader Consulations]\n' + leader_consulations + '\n' + rebuttals + '\n[Panel Leader Final Report]\n' + leader_final_report

    token_usage = count_token_usage(token_usage, panel_leader.get_token_usage())

    return level3_option, level3_report, token_usage

def hierachy_diagnosis(dataset, medical_case):
    original_case = medical_case
    medical_image = medical_case['image']
    medical_case = medical_case['question']

    token_usage = {
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0
        }
    
    # Level-1 diagnosis
    print('\n=====[LEVEL-1] Initial Assessment=====')
    level1_responses = []
    level1_option = []
    level1_reasoning = []
    level1_report_dict = {}

    for i in range(2):
        expert = Agent()
        query = """[Core Identity] You are a professional and rigorous <MEDICAL FIELD> expert specializing in diagnostic imaging interpretation (<IMAGING MODALITIES>). Your core goal is to make precise, evidence-based diagnoses for the given question strictly based on the provided <IMAGING TYPE> image and medical case. [Medical Case] {medical_case}. [Reasoning Requirements] Follow these steps in your reasoning:  Follow these steps in your reasoning: 1. First check the image and read the question carefully. 2. Describe the key visual features observed in the image. 3. Explain the radiological implications of these findings. 4. Conclude which option is the best fit and clarify the rationale. [Strict Output Format] #Reasoning: <3-5 sentences of reasoning> #Answer: <a single letter of your choice, e.g. A or B.>.""".format(medical_case=medical_case)

        output_response = expert.chat(query, medical_image, temperature=0.7)

        print(f'\n[LEVEL-1][Agent{i+1}:{expert.model_info}]\n', output_response)
        option = extract_option(output_response.split(':')[-1])

        level1_responses.append(output_response)
        level1_option.append(option)
        reasoning = output_response
        token_usage = count_token_usage(token_usage, expert.get_token_usage())

        level1_reasoning.append(reasoning)
        if option not in level1_report_dict.keys():
            level1_report_dict[option] = [reasoning]
        else:
            level1_report_dict[option].append(reasoning)

    level1_options = [item for item in level1_report_dict.items()]

    level1_report = "[Level-1 Initial Assessment Reports]\n" + "\n".join([f'<Agent{i+1} #Choice: {option}#Reasoning: {level1_reasoning[i]}>' for i, option in enumerate(level1_option)])
    latest_report = "" + level1_report
    path_flag = 0

    if len(level1_options)==1:
        level2_option, level2_reasoning, token_usage = level_2_diagnosis(dataset, medical_case, medical_image, level1_report, token_usage)
        latest_report = level1_report + '\n[Level-2 Extra Expert Check]\n' + '<' + level2_reasoning + '>'

        if len(level1_options)==1 and level2_option==level1_options[0][0]:
            return level2_option, "level-2", token_usage
        else:
            if level2_option in level1_report_dict.keys():
                level2_dict[level2_option].append(level2_reasoning)
            else:
                level2_dict[level2_option] = level2_reasoning
        path_flag = 1

    # Enter Level-3 diagnosis: panel debate (critc mode)
    level3_option, level3_report, token_usage = level_3_diagnosis(medical_case, medical_image, latest_report, level1_report_dict, token_usage)
    latest_report = latest_report + '\n' + level3_report

    return level3_option, ["level-3","2->3"][path_flag], token_usage
