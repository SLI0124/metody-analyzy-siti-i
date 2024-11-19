# Cvičení

## Cvičení 1

První analýza síťových dat - implementujte vše sami, stačí konzolová aplikace. Data z kolekce Karate
Club reprezentujte maticí sousednosti (adjacency matrix) a seznamem vrcholů a jejich sousedů (adjacency list). Určete
min, max a průměrný stupeň vrcholů. Určete četnost a relativní četnost výskytů vrcholů s daným stupněm a vytvořte
histogram těchto četností (sloupcový graf, kde na ose x je stupeň a na ose y počet vrcholů s daným stupněm).

## Cvičení 2

Zadání cvičení - implementujte vše sami, stačí konzolová aplikace. Pro Karate Club určete vzdálenost (délku nejkratší
cesty) mezi všemi dvojicemi vrcholů

$$e(v_i) = max_j{d(v_i, v_j)}$$

průměrnou vzdálenost:

$$ \frac{2}{n(n-1)} \sum_{i=1}^{n} \sum_{j>i}^{n} d(i,j) $$

a průměr:

$$ d(G) max_i{e(v_i)} = max_i,j{d(v_i, v_j)} $$

Budete potřebovat [Floydův algoritmus](https://www.youtube.com/watch?v=4OQeCuLYj-4). Dále určete closeness centralitu
každého vrcholu:

$$C(i) = \frac{n}{\sum_{j=1}^{n} d(i,j)}$$

## Cvíčení 3

Pro Karate Club určete shlukovací koeficient každého vrcholu a určete tranztivitu sítě (tedy průměrný shlukovací
koeficient). Určete shlukovací efekt. Ten se určí jako průměrný CC pro vrcholy daného stupně. Distribuci tohoto
průměrného CC (osa Y) vůči stupni (osa X) vykreslete (výsledek). Výsledky měření všech lokálních vlastností uložte do
CSV souboru. Ten bude obsahovat následujcí sloupce: ID vrcholu, jeho stupeň, closeness centralitu a shlukovací
koeficient.

## Cvičení 4

Detekce komunitní struktury pro Karate Club. Určete komunitní strukturu těmito algoritmy (metodami):

- louvain_communities()
- label_propagation_communities()
- kernighan_lin_bisection() (najděte 4 komunity)
- girvan_newman() (najděte cca 4 komunity)
- k_clique_communities() (nelze spčítat modularitu)
- Komunitní strukturu pro každý algoritmus vizualizujte.
- Vyhodnoťte modularitu pro komunitní strukturu nalezenou všemi algoritmy (modularity()).
- Dále si k csv souboru z minulého cvičení pro každý algoritmus přidejte sloupce s id komunity, ke které dle příslušného
  algoritmu uzel patří.

## Cvičení 5

- Naimplementujte metodu kNN, e-radius a kombinaci obou metod, kde pro podobnost použijte Gaussian kernel.
- Experimentujte s nastavením parametrů k a e, tak abyste vypozorovali vliv těchto parametrů na vlastnosti výsledných
  sítí. Povinně použijte parametry k=3 a Sigma=1, epsilon=0.9.
- Použijte Gephi pro výpočet základních vlastností a vizualizaci sítí.
- Vytvořte report, kde pro každou metodu a každou testovanou hodnotu parametru / parametrů budete mít
  vizualizaci sítě vzhledem k modularitě a informace o počtu komunit, průměrném stupni a počtu komponent souvislosti.
  Obarvěte vrcholy také vzhledem ke třídě jednotlivých datových instancí (záznamů) v původním datasetu, tj. vzhledem k
  typu kosatce (iris setosa, iris versicolor respektive iris virginica).

## Cvičení 6

- Cvičení - Vygenerujte 3 náhodné grafy (tj. naimplementujte (sami, ne pomocí knihoven ...) generování sítě) s parametry
  n=550 (počet vrcholů) a pravděpodobností p existence hrany, kterou nastavíte tak, aby Vám průměrný stupeň vyšel menší
  než 1 (např. p = 0.001), cca roven 1 (např. p = 0.0059) a větší než 1 (např. p = 0.01).
    - Sítě vyexportujte do formátu vhodného pro Gephi. Určete všechny vlastnosti vygenerovaných sítí (grafů), které v
      tuto chvíli určit umíte (počet komponent souvislosti, distribuce velikosti komponent souvislosti, velikost
      největší komponenty souvislosti, průměr, průměrná vzdálenost (průměrná vzdálenost přes jednotlivé komponenty
      souvislosti), shlukovací koeficient a distribuce stupňů, komunitní struktura, centrality).
    - Sítě vizualizujte a pro globální vlastnosti vytvořte vhodnou tabulku s jejich hodnotami.
    - Jaké rozdíly mezi sítěmi pozorujete vzhledem k pravděpodobnosti p a sledovaným vlastnostem?

## Cvičení 7

Naimplementujte (tj. sami, nesmíte použít žádnou knihovní funkci) model preferenčního připojování (BA model).

- Vygenerujte dvě sítě, jednu pro m = 2, druhou pro m = 3. Cílový stav jsou sítě s 550 vrcholy.
- Sítě vyexportujte do formátu vhodného pro Gephi.
- Určete všechny vlastnosti vygenerovaných sítí (grafů), které v tuto chvíli určit umíte (počet komponent souvislosti,
  distribuce velikosti komponent souvislosti, velikost největší komponenty souvislosti, průměr, průměrná vzdálenost (
  průměrná vzdálenost přes jednotlivé komponenty souvislosti), shlukovací koeficient a distribuce stupňů, komunitní
  struktura, centrality).
- Sítě vizualizujte.
