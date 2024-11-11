# Analýza sítě

(min. 17, max. 34 bodů)

Použít vybrané metody k analýze reálné sítě s alespoň deseti tisíci vrcholy. Síť musí být:

1. **Síťové zdroje**
    - Síť přímo stažená z úložiště (např. [networkrepository.com](https://networkrepository.com/))
    - nebo síť zkonstruovaná z vektorových dat stažených z webů jako [Kaggle](https://www.kaggle.com/)
      či [UCI Machine Learning Repository](https://archive.ics.uci.edu/).

2. **Úloha bude rozdělena do dílčích kroků:**

   ### a) Analýza základních strukturálních vlastností sítě
   Vyjádřená tabulkou (pro globální vlastnosti), grafem distribuce hodnot dané lokální vlastnosti a vizualizací.

    - **Strukturální metriky:**
        - Počet vrcholů, počet hran, hustota sítě
        - Průměrný stupeň, maximální stupeň a distribuce stupňů
        - Centrality
        - Shlukovací koeficient (CC) – průměr. Graf shlukovacího efektu (průměrný CC pro vrcholy daného stupně; osa Y:
          distribuce průměrného CC, osa X: stupeň)
        - Souvislost - počet souvislých komponent a distribuce jejich velikostí
        - Vizualizace sítě (včetně zvýraznění center/hubů)

   ### b) Analýza komunitní struktury
   Použití alespoň dvou různých algoritmů pro detekci komunit.

    - **Komunitní metriky:**
        - Počet komunit, průměrná, minimální a maximální velikost komunit, hodnota modularity
        - Distribuce velikostí komunit
        - Vizualizace podle komunitní struktury

3. **Použité nástroje**
    - R, Python, Gephi nebo jejich kombinace

4. **Výstup**
    - Textový dokument ve formátu PDF, obsahující:
        - Popis datového souboru (kde a jak byl získán, co obsahuje atd.)
        - Výsledky analýzy a interpretaci (co výsledky znamenají)

5. **Deadline**
    - **Termín odevzdání:** 16. 12. 2024
    - **Prezentace výsledků analýzy:** Po písemném testu
    - PDF soubor zaslat na mou emailovou adresu, v předmětu emailu uvést „MAS1 analýza váš_login“. Soubor pojmenujte
      stejně.
