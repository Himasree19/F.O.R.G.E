import matplotlib.pyplot as plt

# ==========================================
# FEATURE CONTRIBUTION GRAPH
# ==========================================

def generate_feature_chart(

    parameter_contribution,

    output_path
):

    labels = []
    scores = []

    for key, value in parameter_contribution.items():

        labels.append(
            key.upper()
        )

        scores.append(
            value["score"]
        )

    plt.figure(figsize=(8,5))

    plt.bar(
        labels,
        scores
    )

    plt.xlabel("Parameters")

    plt.ylabel("Contribution")

    plt.title(
        "Feature Contribution"
    )

    plt.savefig(output_path)

    plt.close()

    return output_path