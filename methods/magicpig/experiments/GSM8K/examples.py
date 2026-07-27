import json


def get_examples():
    examples = {}
    examples["gsm8k-cot"] = [
        (
            "question: There are 15 trees in the grove. Grove workers will plant trees in thegrove today. After they are done, there will be 21 trees. How many trees didthe grove workers plant today?",
            "target: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 15 = 6. The answer is 6.",
        ),
        (
            "question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
            "target: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.",
        ),
        (
            "question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
            "target: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39."
        ),
        (
            "question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12lollipops. How many lollipops did Jason give to Denny?",
            "target: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8."
        ),
        (
            "question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
            "target: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9."
        ),
        (
            "question: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
            "target: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29."
        ),
        ( 
            "question: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
            "target: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33.",
        ),
        (
            "question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
            "target: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 = 8 dollars left. The answer is 8.",
        ),
    ]
    
    examples["gsm8k-simple"] = [
        (
            "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
            "Let's think step by step\nThere are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6.\n<answer>6</answer>.",
        ),
        (
            "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
            "Let's think step by step\nThere are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.\n<answer>5</answer>.",
        ),
        (
            "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
            "Let's think step by step\nOriginally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39.\n<answer>39</answer>.",
        ),
        (
            "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
            "Let's think step by step\nJason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8.\n<answer>8</answer>.",
        ),
        (
            "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
            "Let's think step by step\nShawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9.\n<answer>9</answer>.",
        ),
        (
            "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
            "Let's think step by step\nThere were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29.\n<answer>29</answer>.",
        ),
        (
            "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
            "Let's think step by step\nMichael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls.\n<answer>33</answer>.",
        ),
        (
            "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
            "Let's think step by step\nOlivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8.\n<answer>8</answer>.",
        ),
    ]

    return examples


EXAMPLES = get_examples()

def load_prompt(prompt_name, num_shots):
    if not num_shots:
        return []
    return EXAMPLES[prompt_name][:num_shots]


def construct_prompt(example, tokenizer):
    # return example["question"]
    demos = load_prompt("gsm8k-cot", 8)
    demo_prompt = "".join(
        [
            q + "\n" + a
            for q, a in demos
        ]
    )
    message = demo_prompt + "\nQuestion: " + example["question"] + "\n"

    prompt = [{ "role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}." }, {"role": "user", "content": message }]
    prompt = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False)
    return prompt



