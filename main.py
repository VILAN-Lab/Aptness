import csv
from pathlib import Path


LLM_RESULT_DIR = Path("LLM_result")


def get_new_label_by_aptness_rate(label_num=6):
    import pandas as pd

    LLM_RESULT_DIR.mkdir(parents=True, exist_ok=True)

    # Read the CSV file
    df = pd.read_csv("result.csv")

    # Define the 3-class mapping function
    def map_score_to_3_class(score):
        if 1 <= score < 5:
            return 0
        elif 5 <= score < 7:
            return 1
        elif 7 <= score <= 10:
            return 2
        else:
            return -1  # Exception handling

    # Define the 6-class mapping function
    def map_score_to_6_class(score):
        if 0 <= score <= 4.0:
            return 0  # Class 1
        elif 4.333 < score <= 5.0:
            return 1  # Class 2
        elif 5.0 < score < 6.0:
            return 2  # Class 3
        elif 6.0 <= score < 7.0:
            return 3  # Class 4
        elif 7 <= score <= 7.333:
            return 4  # Class 5
        elif 7.33 < score:
            return 5  # Class 6
        else:
            return -1  # Exception handling

    # Select the mapping function according to the number of classes
    if label_num == 3:
        map_score_to_class = map_score_to_3_class
    elif label_num == 6:
        map_score_to_class = map_score_to_6_class
    else:
        raise ValueError("label_num must be either 3 or 6.")

    # Calculate the proportion of each original score/class value
    counts = df.iloc[:, 11].value_counts(normalize=True).sort_index()
    for label, ratio in counts.items():
        print(f"Class {label} ratio: {ratio:.2%}")

    # Get the score column from column L, which corresponds to index 11
    score_col = df.columns[11]

    # Generate a new classification column
    new_class_col = df[score_col].apply(map_score_to_class)

    # Insert the new column after column L, at index position 12
    df.insert(loc=12, column="score_class", value=new_class_col)

    # Calculate the proportion of each generated class
    counts = df["score_class"].value_counts(normalize=True).sort_index()
    for label, ratio in counts.items():
        print(f"Class {label} ratio: {ratio:.2%}")

    # Save the output file
    output_file = LLM_RESULT_DIR / f"result_labeled_{label_num}.csv"
    df.to_csv(output_file, index=False)

    return output_file


def Run_aptness_accuracy():
    MODEL_NAME_LIST = [
        "glm4-9b",
        "phi4-14b",
        "gemma3-27b",
        "qwq-32b",
        "llava-34b",
        "vicuna-33b",
        "deepseek-r1-8b",
        "deepseek-r1-70b",
        "mixtral-8x7b",
        "llama3.3-70b",
        "gpt4omini",
        "gpt4o",
    ]

    for MODEL_NAME in MODEL_NAME_LIST:

        LABEL_NUM = 3
        print("MODEL:", MODEL_NAME, "LABELS:", LABEL_NUM)
        convert_LLM_answer_to_label(MODEL_NAME)
        LLM_result_label_combine(MODEL_NAME, LABEL_NUM)
        cal_accuracy_by_bin(MODEL_NAME, LABEL_NUM)

        LABEL_NUM = 6
        print("MODEL:", MODEL_NAME, "LABELS:", LABEL_NUM)
        convert_LLM_answer_to_label(MODEL_NAME)
        LLM_result_label_combine(MODEL_NAME, LABEL_NUM)
        cal_accuracy_by_bin(MODEL_NAME, LABEL_NUM)


def convert_LLM_answer_to_label(MODEL_NAME):
    LLM_RESULT_DIR.mkdir(parents=True, exist_ok=True)

    input_file_name = "VUA20_int_" + MODEL_NAME + ".csv"
    output_file_name = LLM_RESULT_DIR / ("VUA20_aptness_" + MODEL_NAME + ".csv")

    ERROR_NUM = 0

    with open(input_file_name, "r", newline="", encoding="utf-8") as input_file, \
            open(output_file_name, "w", newline="", encoding="utf-8") as output_file:

        fr = csv.reader(input_file)
        fw = csv.writer(output_file)

        for i, row in enumerate(fr):

            if not row:
                continue

            answer = row[8]

            if "yes" in answer.lower():
                answer = "Yes"
            elif "no" in answer.lower():
                answer = "No"
            else:
                ERROR_NUM += 1
                answer = "No"

            fw.writerow(row[:-1] + [answer])

    # print("ERROR_NUM:", ERROR_NUM)
    # print(output_file_name)
    return output_file_name


def LLM_result_label_combine(MODEL_NAME, LABEL_NUM):
    import pandas as pd

    LABEL_NUM = str(LABEL_NUM)

    # File paths
    main_file = LLM_RESULT_DIR / ("VUA20_aptness_" + MODEL_NAME + ".csv")
    label_file = LLM_RESULT_DIR / ("result_labeled_" + LABEL_NUM + ".csv")

    # Use integer indexes: Python indexes start from 0
    col_from_idx = 11
    col_to_idx = 6

    # Read the files
    df_main = pd.read_csv(main_file)
    df_label = pd.read_csv(label_file)

    # Set the first column as the index, which represents the sample ID
    df_main.set_index(df_main.columns[0], inplace=True)
    df_label.set_index(df_label.columns[0], inplace=True)

    # Extract the aptness label column from result_labeled.csv
    new_data = df_label.iloc[:, col_from_idx]

    # Write the data into the target column.
    # If the target column does not exist, create a new column.
    if col_to_idx < len(df_main.columns):
        target_col_name = df_main.columns[col_to_idx]
    else:
        target_col_name = f"Inserted_Col_{col_to_idx + 1}"  # Add 1 for human-readable column numbering
        df_main[target_col_name] = None  # Add a new column

    # Align by sample ID and write the data into the target column
    df_main[target_col_name] = new_data

    # Save the result
    df_main.reset_index(inplace=True)
    output_file = LLM_RESULT_DIR / ("VUA20_aptness_" + MODEL_NAME + "_labeled.csv")
    df_main.to_csv(output_file, index=False)

    return output_file


def cal_accuracy_by_bin(MODEL_NAME, num_classes=3):
    import pandas as pd
    import numpy as np

    csv_file = LLM_RESULT_DIR / ("VUA20_aptness_" + MODEL_NAME + "_labeled.csv")

    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Ensure that the CSV file has at least 9 columns.
    # The 8th column is the aptness level, and the 9th column is the predicted label.
    if df.shape[1] < 9:
        raise ValueError("The CSV file has fewer than 9 columns.")

    # Extract aptness levels from the 8th column and predicted labels from the 9th column
    similarity = df.iloc[:, 7]  # Aptness level column
    labels = df.iloc[:, 8]      # Predicted label column

    # Select valid aptness levels according to the number of classes
    if num_classes == 3:
        valid_levels = [0, 1, 2]
    elif num_classes == 6:
        valid_levels = [0, 1, 2, 3, 4, 5]
    else:
        raise ValueError("num_classes must be either 3 or 6.")

    # Keep only valid rows
    valid_indices = similarity.isin(valid_levels)
    similarity = similarity[valid_indices]
    labels = labels[valid_indices]

    # Calculate the accuracy and sample count for each level
    accuracy_results = []
    for level in valid_levels:
        level_mask = similarity == level
        level_count = np.sum(level_mask)

        if level_count == 0:
            accuracy = None
        else:
            accuracy = np.mean(labels[level_mask] == "Yes")  # The label is the string "Yes"

        accuracy_results.append((level, accuracy, level_count))

    # Convert the results to a DataFrame and print them
    result_df = pd.DataFrame(
        accuracy_results,
        columns=["Level", "Accuracy", "Sample Count"]
    )

    print(result_df)
    return result_df