import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
BASE_N_SPLITS = 10

# ----------------------------
# 1) Wczytywanie i preprocessing
# ----------------------------


def load_and_preprocess_breast_cancer():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    features = list(data.feature_names)
    return df, features, "target"


def load_and_preprocess_mushroom():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
    column_names = [
        "target",
        "cap-shape",
        "cap-surface",
        "cap-color",
        "bruises",
        "odor",
        "gill-attachment",
        "gill-spacing",
        "gill-size",
        "gill-color",
        "stalk-shape",
        "stalk-root",
        "stalk-surface-above-ring",
        "stalk-surface-below-ring",
        "stalk-color-above-ring",
        "stalk-color-below-ring",
        "veil-type",
        "veil-color",
        "ring-number",
        "ring-type",
        "spore-print-color",
        "population",
        "habitat",
    ]
    df = pd.read_csv(url, header=None, names=column_names, na_values="?")
    df["stalk-root"] = df["stalk-root"].fillna(df["stalk-root"].mode()[0])

    for col in df.columns:
        df[col] = LabelEncoder().fit_transform(df[col])

    features = [c for c in df.columns if c != "target"]
    return df, features, "target"


def load_and_preprocess_adult():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    column_names = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "target",
    ]
    df = pd.read_csv(
        url, header=None, names=column_names, na_values=" ?", skipinitialspace=True
    )
    df = df.dropna()

    df = df.drop(columns=["fnlwgt", "education"])
    df["target"] = (df["target"].astype(str).str.strip() == ">50K").astype(int)

    features = [c for c in df.columns if c != "target"]
    return df, features, "target"


def load_and_preprocess_car():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
    column_names = [
        "buying",
        "maint",
        "doors",
        "persons",
        "lug_boot",
        "safety",
        "target",
    ]
    df = pd.read_csv(url, header=None, names=column_names)

    for col in df.columns:
        df[col] = LabelEncoder().fit_transform(df[col])

    features = [c for c in df.columns if c != "target"]
    return df, features, "target"


# ----------------------------
# 2) Wprowadzanie niespójności
# ----------------------------


def inject_inconsistency(dataset_name, df, features, target):
    df_mod = df.copy()
    info = {"dropped_columns": [], "transform": ""}

    if dataset_name == "Breast Cancer":
        keep = ["mean radius", "mean texture", "mean area"]
        drop = [c for c in features if c not in keep]
        df_mod = df_mod.drop(columns=drop)
        info["dropped_columns"] = drop

        for col in keep:
            df_mod[col] = pd.qcut(df_mod[col], q=4, labels=False, duplicates="drop")

        info["transform"] = (
            "Zredukowano do 3 cech i zastosowano dyskretyzację (qcut, 4 przedziały)."
        )
        features_mod = keep

    elif dataset_name == "Mushroom":
        keep = ["cap-color", "gill-size", "habitat", "bruises"]
        keep = [c for c in keep if c in df_mod.columns]
        drop = [c for c in df_mod.columns if c not in keep + [target]]
        df_mod = df_mod.drop(columns=drop)
        info["dropped_columns"] = drop

        for col in keep:
            df_mod[col] = (df_mod[col] % 3).astype(int)

        info["transform"] = (
            "Zredukowano do 4 cech i połączono kategorie (mapowanie mod 3)."
        )
        features_mod = keep

    elif dataset_name == "Adult":
        df_mod["age_bin"] = (df_mod["age"] // 10).astype(int)
        df_mod["hours_bin"] = (df_mod["hours-per-week"] // 10).astype(int)

        keep = [
            "age_bin",
            "education-num",
            "sex",
            "hours_bin",
            "workclass",
            "marital-status",
        ]
        drop = [c for c in df_mod.columns if c not in keep + [target]]
        df_mod = df_mod.drop(columns=drop)
        info["dropped_columns"] = drop

        cat_cols = df_mod.select_dtypes(include=["object"]).columns
        df_mod = pd.get_dummies(df_mod, columns=cat_cols, drop_first=True)

        features_mod = [c for c in df_mod.columns if c != target]
        info["transform"] = (
            "Utworzono age_bin i hours_bin, zredukowano cechy, wykonano one-hot encoding."
        )

    elif dataset_name == "Car":
        # KLUCZOWA ZMIANA:
        # - usuwamy tylko 1 kolumnę (buying) zamiast 2
        # - łączenie kategorii bardzo łagodne: mod 6
        # -> niespójność nadal będzie, ale drastyczny ma szansę zachować kilka klas
        drop = ["buying"]
        df_mod = df_mod.drop(columns=drop)
        info["dropped_columns"] = drop

        for col in [c for c in df_mod.columns if c != target]:
            df_mod[col] = (df_mod[col] % 6).astype(int)

        features_mod = [c for c in df_mod.columns if c != target]
        info["transform"] = (
            "Usunięto 1 cechę (buying) oraz łagodnie połączono kategorie (mod 6), aby utrzymać wariant drastyczny."
        )

    else:
        features_mod = features

    return df_mod, features_mod, info


# ----------------------------
# 3) Niespójności + warianty
# ----------------------------


def find_inconsistent_blocks(df, feature_names, target_name):
    dup = df.duplicated(subset=feature_names, keep=False)
    if not dup.any():
        return pd.Index([]), pd.Index([]), 0, 0

    df_dup = df[dup]
    inconsistent = df_dup.groupby(feature_names).filter(
        lambda x: x[target_name].nunique() > 1
    )
    if inconsistent.empty:
        return pd.Index([]), pd.Index([]), 0, 0

    idx_quant = []
    for _, g in inconsistent.groupby(feature_names):
        maj = g[target_name].mode()[0]
        idx_quant.extend(g[g[target_name] != maj].index)

    idx_drastic = inconsistent.index
    n_blocks = inconsistent.groupby(feature_names).ngroups
    n_rows = len(inconsistent)

    return pd.Index(idx_quant), idx_drastic, n_blocks, n_rows


def get_data_variants(df, feature_names, target_name):
    idx_q, idx_d, n_blocks, n_rows = find_inconsistent_blocks(
        df, feature_names, target_name
    )

    variants = {
        "1. Bazowy": df.copy(),
        "2. Ilościowy": df.drop(index=idx_q),
        "3. Drastyczny": df.drop(index=idx_d),
    }
    stats = {
        "n_blocks": n_blocks,
        "n_rows": n_rows,
        "removed_quant": len(idx_q),
        "removed_drastic": len(idx_d),
    }
    return variants, stats


# ----------------------------
# 4) Eksperyment
# ----------------------------


def choose_n_splits(y, preferred=BASE_N_SPLITS):
    """Dobiera liczbę foldów do najmniejszej liczności klasy (żeby StratifiedKFold działał)."""
    counts = pd.Series(y).value_counts()
    min_class = int(counts.min())
    # żeby CV było sensowne: co najmniej 2 foldy, ale nie więcej niż min_class
    return max(2, min(preferred, min_class))


def run_experiment(dataset_name, df_loader):
    print(f"\n--- Rozpoczynanie eksperymentu dla: {dataset_name} ---")

    df, features, target = df_loader()
    print(f"Wczytano {len(df)} wierszy, {len(features)} atrybutów (przed modyfikacją).")

    df_mod, features_mod, info = inject_inconsistency(
        dataset_name, df, features, target
    )
    print(f"Po modyfikacji: {len(df_mod)} wierszy, {len(features_mod)} atrybutów.")
    if info["dropped_columns"]:
        print(
            f"  Usunięte kolumny: {', '.join(info['dropped_columns'][:10])}{'...' if len(info['dropped_columns'])>10 else ''}"
        )
    if info["transform"]:
        print(f"  Transformacja: {info['transform']}")

    variants, inc_stats = get_data_variants(df_mod, features_mod, target)
    print(
        f"Wykryto niespójność: bloki={inc_stats['n_blocks']}, wiersze_w_blokach={inc_stats['n_rows']}. "
        f"Do usunięcia: ilościowa={inc_stats['removed_quant']}, drastyczna={inc_stats['removed_drastic']}"
    )

    results = []

    models = {
        "Drzewo Decyzyjne": DecisionTreeClassifier(
            criterion="entropy", random_state=RANDOM_STATE
        ),
        "Las Losowy": RandomForestClassifier(
            n_estimators=100, criterion="entropy", random_state=RANDOM_STATE
        ),
    }

    for variant_name, df_variant in variants.items():
        if df_variant.empty or df_variant[target].nunique() < 2:
            print(f"Pominięto wariant '{variant_name}' (za mało danych lub 1 klasa).")
            continue

        X = df_variant[features_mod]
        y = df_variant[target]

        n_splits = choose_n_splits(y, preferred=BASE_N_SPLITS)
        skf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE
        )

        for model_name, model in models.items():
            fold_acc = []
            fold_depth = []
            fold_nodes = []

            for tr, te in skf.split(X, y):
                clf = model.fit(X.iloc[tr], y.iloc[tr])
                y_pred = clf.predict(X.iloc[te])
                fold_acc.append(accuracy_score(y.iloc[te], y_pred))

                if model_name == "Drzewo Decyzyjne":
                    fold_depth.append(clf.get_depth())
                    fold_nodes.append(clf.tree_.node_count)

            acc_mean = float(np.mean(fold_acc))
            acc_std = float(np.std(fold_acc, ddof=1)) if len(fold_acc) > 1 else 0.0

            results.append(
                {
                    "Zbiór": dataset_name,
                    "Wariant": variant_name,
                    "Model": model_name,
                    "CV_folds": n_splits,
                    "Liczba wierszy": len(df_variant),
                    "Accuracy_mean": acc_mean,
                    "Accuracy_std": acc_std,
                    "Śr. Głębokość": float(np.mean(fold_depth)) if fold_depth else None,
                    "Śr. Liczba Węzłów": (
                        float(np.mean(fold_nodes)) if fold_nodes else None
                    ),
                    "Bloki niespójne": inc_stats["n_blocks"],
                    "Wiersze w niespójności": inc_stats["n_rows"],
                }
            )

    return results


# ----------------------------
# 5) Main + wyniki + LaTeX
# ----------------------------


def main():
    datasets_to_run = {
        "Breast Cancer": load_and_preprocess_breast_cancer,
        "Mushroom": load_and_preprocess_mushroom,
        "Adult": load_and_preprocess_adult,
        "Car": load_and_preprocess_car,
    }

    all_results = []
    for name, loader in datasets_to_run.items():
        all_results.extend(run_experiment(name, loader))

    df = pd.DataFrame(all_results)

    df["Dokładność"] = df.apply(
        lambda r: f"{r['Accuracy_mean']*100:.2f} ± {r['Accuracy_std']*100:.2f}%", axis=1
    )
    df["Śr. Głębokość"] = df["Śr. Głębokość"].map(
        lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"
    )
    df["Śr. Liczba Węzłów"] = df["Śr. Liczba Węzłów"].map(
        lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"
    )

    print("\n\n--- ZBIORCZE WYNIKI ---")
    cols = [
        "Zbiór",
        "Wariant",
        "Model",
        "CV_folds",
        "Liczba wierszy",
        "Dokładność",
        "Śr. Głębokość",
        "Śr. Liczba Węzłów",
        "Bloki niespójne",
        "Wiersze w niespójności",
    ]
    print(df[cols].to_string(index=False))

    print("\n\n--- KOD LATEX ---")
    df_latex = df.copy()
    df_latex["Wariant"] = df_latex["Wariant"].str.replace(r"^\d+\. ", "", regex=True)
    df_latex["Model"] = df_latex["Model"].str.replace("Drzewo Decyzyjne", "Drzewo Dec.")
    df_latex = df_latex[
        [
            "Zbiór",
            "Wariant",
            "Model",
            "CV_folds",
            "Dokładność",
            "Śr. Głębokość",
            "Śr. Liczba Węzłów",
            "Bloki niespójne",
            "Wiersze w niespójności",
        ]
    ]
    df_latex = df_latex.set_index(["Zbiór", "Wariant", "Model"]).sort_index()

    latex_string = df_latex.to_latex(
        longtable=True,
        caption="Wyniki eksperymentów (wariant bazowy vs metody usuwania niespójności). Dokładność: średnia ± odchylenie standardowe w walidacji krzyżowej (liczba foldów dobierana do rozkładu klas po czyszczeniu).",
        label="tab:wyniki_zbiorcze",
        multicolumn_format="c",
        multirow=True,
    )
    print(latex_string)


if __name__ == "__main__":
    main()
