import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings

# Ignorowanie ostrzeżeń, aby wyniki były czytelne
warnings.filterwarnings("ignore")

# --- 1. Funkcje do wczytywania i przetwarzania danych z UCI ---


def load_and_preprocess_breast_cancer():
    """Wczytuje i przygotowuje zbiór Breast Cancer."""
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    return df, data.feature_names, "target"


def load_and_preprocess_mushroom():
    """Wczytuje i przygotowuje zbiór Mushroom."""
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

    # Metoda usuwania niespójności 1: Wstępne czyszczenie brakujących danych
    # W tym zbiorze, 'stalk-root' ma dużo braków. Zastąpimy je najczęściej występującą wartością (modą).
    df["stalk-root"] = df["stalk-root"].fillna(df["stalk-root"].mode()[0])

    # Kodowanie wszystkich atrybutów kategorycznych na numeryczne
    encoders = {}
    for col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    feature_names = [col for col in df.columns if col != "target"]
    return df, feature_names, "target"


def load_and_preprocess_adult():
    """Wczytuje i przygotowuje zbiór Adult (Census)."""
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

    # Usuwamy wiersze z jakimikolwiek brakami danych (dla uproszczenia)
    df = df.dropna()

    # Usuwamy nieistotną kolumnę
    df = df.drop(
        columns=["fnlwgt", "education"]
    )  # education-num jest numerycznym odpowiednikiem

    # Kodowanie atrybutów kategorycznych
    categorical_cols = df.select_dtypes(include=["object"]).columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Zmieniamy nazwę kolumny docelowej (jeśli istnieje po dummifikacji)
    if "target_>50K" in df.columns:
        df = df.rename(columns={"target_>50K": "target"})

    feature_names = [col for col in df.columns if col != "target"]
    return df, feature_names, "target"


def load_and_preprocess_car():
    """Wczytuje i przygotowuje zbiór Car Evaluation."""
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

    # Kodowanie wszystkich atrybutów
    encoders = {}
    for col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    feature_names = [col for col in df.columns if col != "target"]
    return df, feature_names, "target"


# --- 2. Funkcje do obsługi niespójności ---


def find_inconsistent_blocks(df, feature_names, target_name):
    """Znajduje bloki niespójnych obiektów i zwraca indeksy do usunięcia."""
    # Znajdź wszystkie duplikaty atrybutów warunkowych
    duplicated_features = df.duplicated(subset=feature_names, keep=False)

    if not duplicated_features.any():
        # Brak jakichkolwiek duplikatów cech -> brak niespójności
        return pd.Index([]), pd.Index([])

    # DataFrame zawierający tylko te wiersze, które mają duplikaty cech
    df_duplicates = df[duplicated_features]

    # Grupuj po cechach i sprawdź, czy w grupie jest więcej niż 1 unikalna decyzja
    # To jest nasz blok niespójny
    inconsistent_groups = df_duplicates.groupby(feature_names).filter(
        lambda x: x[target_name].nunique() > 1
    )

    if inconsistent_groups.empty:
        # Istnieją duplikaty, ale są spójne
        return pd.Index([]), pd.Index([])

    # --- Metoda Ilościowa ---
    # Dla każdej niespójnej grupy usuwamy wiersze z decyzją mniejszościową
    indices_to_remove_quantitative = []

    # Grupujemy po cechach warunkowych
    for _, group in inconsistent_groups.groupby(feature_names):
        # Znajdź decyzję większościową w bloku
        majority_decision = group[target_name].mode()[0]

        # Zidentyfikuj indeksy wierszy, które mają decyzję inną niż większościowa
        minority_indices = group[group[target_name] != majority_decision].index
        indices_to_remove_quantitative.extend(minority_indices)

    # --- Metoda Jakościowa/Drastyczna ---
    # Usuwamy wszystkie wiersze, które brały udział w niespójności
    indices_to_remove_drastic = inconsistent_groups.index

    return pd.Index(indices_to_remove_quantitative), indices_to_remove_drastic


def get_data_variants(df, feature_names, target_name):
    """Zwraca 3 warianty DataFrame: Bazowy, Ilościowy, Drastyczny"""

    idx_quant, idx_drastic = find_inconsistent_blocks(df, feature_names, target_name)

    df_bazowy = df.copy()

    # Wariant 2: Usuń tylko mniejszość z niespójnych bloków
    df_ilosciowy = df.drop(index=idx_quant)

    # Wariant 3: Usuń wszystkie obiekty z niespójnych bloków
    df_drastyczny = df.drop(index=idx_drastic)

    variants = {
        "1. Bazowy": df_bazowy,
        "2. Ilościowy": df_ilosciowy,
        "3. Drastyczny": df_drastyczny,
    }

    return variants


# --- 3. Funkcja do przeprowadzania eksperymentu ---


def run_experiment(dataset_name, df_loader):
    """
    Uruchamia pełen eksperyment dla jednego zbioru danych:
    1. Wczytuje dane
    2. Tworzy 3 warianty niespójności
    3. Przeprowadza 10-krotną walidację krzyżową dla Drzewa i Lasu
    4. Zwraca wyniki
    """

    print(f"\n--- Rozpoczynanie eksperymentu dla: {dataset_name} ---")

    try:
        df, features, target = df_loader()
    except Exception as e:
        print(f"BŁĄD: Nie można wczytać lub przetworzyć zbioru {dataset_name}. {e}")
        return []

    print(f"Wczytano {len(df)} wierszy, {len(features)} atrybutów.")

    # Tworzenie wariantów danych
    try:
        data_variants = get_data_variants(df, features, target)
    except Exception as e:
        print(f"BŁĄD: Nie można utworzyć wariantów danych dla {dataset_name}. {e}")
        return []

    results_list = []

    for variant_name, df_variant in data_variants.items():

        if df_variant.empty or df_variant[target].nunique() < 2:
            print(
                f"Pominięto wariant '{variant_name}' (za mało danych po czyszczeniu)."
            )
            continue

        print(
            f"  Testowanie wariantu: {variant_name} (Liczba wierszy: {len(df_variant)})"
        )

        X = df_variant[features]
        y = df_variant[target]

        # Definiowanie modeli
        models = {
            "Drzewo Decyzyjne": DecisionTreeClassifier(
                criterion="entropy", random_state=42
            ),
            "Las Losowy": RandomForestClassifier(
                n_estimators=100, criterion="entropy", random_state=42
            ),
        }

        # Walidacja krzyżowa
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

        for model_name, model in models.items():

            fold_accuracies = []
            fold_depths = []
            fold_nodes = []

            # Ręczna pętla walidacji krzyżowej, aby zebrać metryki drzewa
            for train_index, test_index in skf.split(X, y):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                # Trenowanie
                clf = model.fit(X_train, y_train)

                # Predykcja i dokładność
                y_pred = clf.predict(X_test)
                fold_accuracies.append(accuracy_score(y_test, y_pred))

                # Zbieranie metryk tylko dla Drzewa Decyzyjnego
                if model_name == "Drzewo Decyzyjne":
                    fold_depths.append(clf.get_depth())
                    fold_nodes.append(clf.tree_.node_count)

            # Uśrednianie wyników
            avg_accuracy = np.mean(fold_accuracies)
            avg_depth = np.mean(fold_depths) if fold_depths else None
            avg_nodes = np.mean(fold_nodes) if fold_nodes else None

            # Zapisz wynik
            results_list.append(
                {
                    "Zbiór danych": dataset_name,
                    "Wariant": variant_name,
                    "Model": model_name,
                    "Liczba wierszy": len(df_variant),
                    "Dokładność (CV=10)": avg_accuracy,
                    "Śr. Głębokość": avg_depth,
                    "Śr. Liczba Węzłów": avg_nodes,
                }
            )

    return results_list


# --- 4. Główna pętla programu ---


def main():

    datasets_to_run = {
        "Breast Cancer": load_and_preprocess_breast_cancer,
        "Mushroom": load_and_preprocess_mushroom,
        "Adult (Census)": load_and_preprocess_adult,
        "Car Evaluation": load_and_preprocess_car,
    }

    all_results = []

    for name, loader_func in datasets_to_run.items():
        all_results.extend(run_experiment(name, loader_func))

    # Prezentacja wyników
    df_results = pd.DataFrame(all_results)

    # Formatowanie dla czytelności
    df_results["Dokładność (CV=10)"] = df_results["Dokładność (CV=10)"].map(
        lambda x: f"{x*100:.2f}%"
    )
    df_results["Śr. Głębokość"] = df_results["Śr. Głębokość"].map(
        lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"
    )
    df_results["Śr. Liczba Węzłów"] = df_results["Śr. Liczba Węzłów"].map(
        lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"
    )

    print("\n\n--- ZBIORCZE WYNIKI EKSPERYMENTÓW ---")
    print(df_results.to_string(index=False))

    # Generowanie kodu LaTeX dla tabeli do wklejenia do pracy
    print("\n\n--- KOD LATEX DLA TABELI WYNIKÓW ---")
    df_latex = df_results.copy()
    # Czyszczenie nazw dla LaTeX
    df_latex["Wariant"] = df_latex["Wariant"].str.replace(r"^\d+\. ", "", regex=True)
    df_latex["Model"] = df_latex["Model"].str.replace("Drzewo Decyzyjne", "Drzewo Dec.")
    df_latex = df_latex.rename(
        columns={
            "Zbiór danych": "Zbiór",
            "Dokładność (CV=10)": "Dokładność",
            "Śr. Głębokość": "Głębokość",
            "Śr. Liczba Węzłów": "Węzły",
        }
    )

    # Ustawienie wielopoziomowego indeksu do grupowania
    df_latex = df_latex.set_index(["Zbiór", "Wariant", "Model"]).sort_index()

    latex_string = df_latex.to_latex(
        longtable=True,
        caption="Zbiorcze wyniki eksperymentów porównujących wpływ metod usuwania niespójności na klasyfikatory.",
        label="tab:wyniki_zbiorcze",
        multicolumn_format="c",
        multirow=True,
    )
    print(latex_string)


if __name__ == "__main__":
    main()
