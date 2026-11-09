import random
from collections import defaultdict
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="GraphBridge AI",
    page_icon="🔗",
    layout="wide"
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

@st.cache_data
def generate_data():
    # Four interest communities, with some cross-community interests.
    domains = {
        "AI & Data": [
            "AI Workshop", "Data Science Lab",
            "Machine Learning Challenge", "NLP Bootcamp",
            "Computer Vision Sprint", "AI Ethics Forum",
            "Data Hackathon", "Python Research Group",
            "Deep Learning Seminar", "Analytics Challenge"
        ],
        "Robotics & Engineering": [
            "Robotics Competition", "Drone Building Lab",
            "IoT Innovation Challenge", "Embedded Systems Club",
            "Automation Workshop", "3D Printing Lab",
            "Smart Devices Sprint", "Mechatronics Forum",
            "Hardware Hackathon", "Robotics Research Group"
        ],
        "Environment & Social Impact": [
            "Climate Data Challenge", "Green Campus Project",
            "Water Quality Lab", "Sustainable Design Sprint",
            "Urban Farming Club", "Energy Innovation Forum",
            "Waste Mapping Project", "EcoTech Hackathon",
            "Biodiversity Survey", "Clean Mobility Challenge"
        ],
        "Design & Entrepreneurship": [
            "Product Design Studio", "Startup Pitch Day",
            "UX Research Workshop", "Social Enterprise Lab",
            "Design Thinking Sprint", "Business Analytics Forum",
            "Innovation Incubator", "Digital Product Challenge",
            "Entrepreneurship Club", "Impact Venture Challenge"
        ]
    }

    item_domain = {
        item: domain
        for domain, items in domains.items()
        for item in items
    }

    users = [f"U{i:03d}" for i in range(1, 101)]
    records = []

    # Each user has a primary interest and may explore other domains.
    domain_names = list(domains.keys())

    for user in users:
        primary = random.choice(domain_names)

        for item in domains[primary]:
            if random.random() < 0.48:
                records.append({
                    "user_id": user,
                    "item": item,
                    "domain": primary
                })

        # Add some cross-domain interests.
        for other_domain in domain_names:
            if other_domain == primary:
                continue

            if random.random() < 0.35:
                item = random.choice(domains[other_domain])
                records.append({
                    "user_id": user,
                    "item": item,
                    "domain": other_domain
                })

        # Occasionally add another random opportunity.
        if random.random() < 0.60:
            item = random.choice(sum(list(domains.values()), []))
            records.append({
                "user_id": user,
                "item": item,
                "domain": item_domain[item]
            })

    df = pd.DataFrame(records).drop_duplicates(
        subset=["user_id", "item"]
    )

    # The resulting interaction count should exceed 300.
    return df, item_domain


df, item_domain = generate_data()

# --------------------------------------------------
# 2. BUILD THE BIPARTITE GRAPH
# --------------------------------------------------

def build_graph(data):
    graph = nx.Graph()

    for row in data.itertuples(index=False):
        user_node = "USER_" + row.user_id
        item_node = "ITEM_" + row.item

        graph.add_node(
            user_node, node_type="user"
        )
        graph.add_node(
            item_node,
            node_type="item",
            domain=row.domain
        )

        graph.add_edge(user_node, item_node)

    return graph


G = build_graph(df)

def user_items(data):
    result = defaultdict(set)

    for row in data.itertuples(index=False):
        result[row.user_id].add(row.item)

    return result


# --------------------------------------------------
# 3. GRAPH-BASED RECOMMENDATIONS
# --------------------------------------------------

def recommend_graph(data, target_user, top_n=5):
    interactions = user_items(data)
    seen = interactions.get(target_user, set())

    if not seen:
        return pd.DataFrame(
            columns=["item", "score", "support"]
        )

    # Build an inverted index:
    # each item -> users who interacted with it.
    item_users = defaultdict(set)

    for user, items in interactions.items():
        for item in items:
            item_users[item].add(user)

    # Weighted support for each candidate item.
    scores = defaultdict(float)
    support = defaultdict(set)

    for other_user, other_items in interactions.items():
        if other_user == target_user:
            continue

        shared = seen & other_items
        union = seen | other_items

        if not union:
            continue

        similarity = len(shared) / len(union)

        if similarity == 0:
            continue

        # Graph path:
        # target user -> shared item -> other user
        # -> candidate item
        for candidate in other_items - seen:
            scores[candidate] += similarity
            support[candidate].add(other_user)

    rows = [
        {
            "item": item,
            "score": round(score, 4),
            "support": len(support[item]),
            "domain": item_domain[item]
        }
        for item, score in scores.items()
    ]

    result = pd.DataFrame(rows)

    if result.empty:
        return result

    # Stable ordering for reproducible demonstrations.
    return result.sort_values(
        ["score", "support", "item"],
        ascending=[False, False, True]
    ).head(top_n)


def recommend_popular(data, target_user, top_n=5):
    seen = user_items(data).get(target_user, set())

    popularity = data.groupby("item")["user_id"].nunique()

    rows = [
        {"item": item, "score": int(count)}
        for item, count in popularity.items()
        if item not in seen
    ]

    return pd.DataFrame(rows).sort_values(
        ["score", "item"],
        ascending=[False, True]
    ).head(top_n)


# --------------------------------------------------
# 4. EXPERIMENT: HIDE ONE KNOWN INTERACTION
# --------------------------------------------------

def leave_one_out_experiment(data, trials=150):
    # Use users with at least two interactions.
    interactions = user_items(data)
    eligible = [
        user for user, items in interactions.items()
        if len(items) >= 2
    ]

    if not eligible:
        return pd.DataFrame()

    rng = random.Random(SEED)
    trials = min(trials, len(eligible))

    selected_users = rng.sample(eligible, trials)
    results = []

    for user in selected_users:
        items = sorted(interactions[user])
        held_out = rng.choice(items)

        # Remove exactly one interaction for evaluation.
        train = data[
            ~(
                (data["user_id"] == user)
                & (data["item"] == held_out)
            )
        ]

        graph_recs = recommend_graph(
            train, user, top_n=5
        )
        popular_recs = recommend_popular(
            train, user, top_n=5
        )

        results.append({
            "user": user,
            "held_out_item": held_out,
            "graph_hit": int(
                held_out in graph_recs["item"].tolist()
            ),
            "popular_hit": int(
                held_out in popular_recs["item"].tolist()
            )
        })

    return pd.DataFrame(results)


# --------------------------------------------------
# 5. EXPERIMENT: VARY NETWORK DENSITY
# --------------------------------------------------

def density_experiment(data):
    rng = np.random.default_rng(SEED)
    results = []

    for keep_rate in [0.35, 0.55, 0.75, 1.0]:
        if keep_rate == 1.0:
            sample = data.copy()
        else:
            mask = rng.random(len(data)) < keep_rate
            sample = data.loc[mask].copy()

        graph = build_graph(sample)

        results.append({
            "Interaction retention": keep_rate,
            "Records": len(sample),
            "Nodes": graph.number_of_nodes(),
            "Edges": graph.number_of_edges(),
            "Connected components": nx.number_connected_components(
                graph
            )
        })

    return pd.DataFrame(results)


# --------------------------------------------------
# 6. STREAMLIT USER INTERFACE
# --------------------------------------------------

st.title("🔗 GraphBridge AI")
st.subheader(
    "Discover hidden opportunities through graph connections"
)

st.write(
    "An educational prototype using bipartite graphs, "
    "Jaccard similarity and indirect graph paths."
)

st.caption(
    "All interaction data is synthetically generated. "
    "It does not represent real students."
)

users = sorted(df["user_id"].unique())

selected_user = st.selectbox(
    "Choose a student",
    users
)

recommendations = recommend_graph(
    df, selected_user, top_n=8
)

user_interaction = df[df["user_id"] == selected_user]

c1, c2, c3, c4 = st.columns(4)

c1.metric("Interaction records", len(df))
c2.metric("Students", df["user_id"].nunique())
c3.metric("Opportunities", df["item"].nunique())
c4.metric("Graph edges", G.number_of_edges())

tab1, tab2, tab3 = st.tabs([
    "Discover",
    "Explore Network",
    "Experiments"
])

with tab1:
    st.header("Personalized opportunity discovery")

    st.write("Existing interests:")
    st.write(", ".join(sorted(user_interaction["item"].unique())))

    if recommendations.empty:
        st.warning(
            "Not enough overlapping interactions for a "
            "graph-based recommendation."
        )
    else:
        st.write(
            "Ranked opportunities the student has not "
            "interacted with:"
        )

        st.dataframe(
            recommendations[
                ["item", "domain", "score", "support"]
            ],
            use_container_width=True,
            hide_index=True
        )

        chosen_item = st.selectbox(
            "Explain a recommendation",
            recommendations["item"].tolist()
        )

        row = recommendations[
            recommendations["item"] == chosen_item
        ].iloc[0]

        st.info(
            f"{chosen_item} belongs to the {row['domain']} "
            f"domain. Its graph score is {row['score']:.4f}. "
            f"It is supported by {row['support']} other "
            "student connection(s)."
        )

        st.caption(
            "The score is a relative ranking signal, not "
            "a probability of success."
        )

    baseline = recommend_popular(df, selected_user, top_n=5)

    st.subheader("Popularity baseline")

    st.dataframe(
        baseline,
        use_container_width=True,
        hide_index=True
    )

with tab2:
    st.header("Student's local network")

    local = nx.Graph()
    user_node = "USER_" + selected_user
    local.add_node(user_node, kind="user")

    for item in user_interaction["item"].unique():
        item_node = "ITEM_" + item
        local.add_node(item_node, kind="item")
        local.add_edge(user_node, item_node)

    # Include one supporting student and their other items.
    if not recommendations.empty:
        selected_rec = recommendations.iloc[0]["item"]

        other_users = df[
            df["item"] == selected_rec
        ]["user_id"].unique()

        for other in other_users[:3]:
            if other == selected_user:
                continue

            other_node = "USER_" + other
            local.add_node(other_node, kind="user")

            for item in user_items(df).get(other, set()):
                item_node = "ITEM_" + item
                local.add_node(item_node, kind="item")
                local.add_edge(other_node, item_node)

    positions = nx.spring_layout(local, seed=SEED)
    colors = [
        "#4C78A8" if local.nodes[n]["kind"] == "user"
        else "#54A24B"
        for n in local.nodes()
    ]

    fig, ax = plt.subplots(figsize=(12, 7))

    nx.draw_networkx(
        local,
        pos=positions,
        ax=ax,
        node_color=colors,
        node_size=850,
        font_size=7,
        width=1.3,
        edge_color="gray"
    )

    ax.set_title("Local bipartite interaction graph")
    ax.axis("off")
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "Blue nodes represent students; green nodes "
        "represent opportunities."
    )

with tab3:
    st.header("Experimental analysis")

    st.subheader("Experiment 1: Recommendation hit rate")

    experiment = leave_one_out_experiment(df)

    if not experiment.empty:
        hit_results = pd.DataFrame({
            "Method": [
                "Graph-based",
                "Popularity baseline"
            ],
            "Hit rate at 5": [
                experiment["graph_hit"].mean(),
                experiment["popular_hit"].mean()
            ]
        })

        st.dataframe(
            hit_results,
            use_container_width=True,
            hide_index=True
        )

        st.bar_chart(
            hit_results.set_index("Method")
        )

        st.write(
            f"Evaluation trials: {len(experiment)}. "
            "A hit means the hidden interaction appears "
            "in the top five recommendations."
        )

        st.caption(
            "This is a single held-out interaction per "
            "eligible user. Results are specific to the "
            "synthetic dataset and random seed."
        )

    st.subheader("Experiment 2: Change graph density")

    density = density_experiment(df)

    st.dataframe(
        density,
        use_container_width=True,
        hide_index=True
    )

    st.line_chart(
        density.set_index("Interaction retention")[
            ["Records", "Edges", "Connected components"]
        ]
    )

    st.caption(
        "Fewer interactions can reduce connectivity and "
        "available recommendation evidence. Interpret "
        "the component counts together with the graph."
    )

    st.subheader("Export the synthetic dataset")

    st.download_button(
        "Download interactions.csv",
        data=df.to_csv(index=False),
        file_name="interactions.csv",
        mime="text/csv"
    )
