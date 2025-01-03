# Cvičení

## Cvičení 1

První analýza síťových dat - implementujte vše sami, stačí konzolová aplikace. Data z kolekce Karate
Club reprezentujte maticí sousednosti (adjacency matrix) a seznamem vrcholů a jejich sousedů (adjacency list). Určete
min, max a průměrný stupeň vrcholů. Určete četnost a relativní četnost výskytů vrcholů s daným stupněm a vytvořte
histogram těchto četností (sloupcový graf, kde na ose x je stupeň a na ose y počet vrcholů s daným stupněm).

## Cvičení 2

Zadání cvičení - implementujte vše sami, stačí konzolová aplikace. Pro Karate Club určete vzdálenost (délku nejkratší
cesty) mezi všemi dvojicemi vrcholů

$$ e(v_i) = \max_{j} \{ d(v_i, v_j) \} $$

průměrnou vzdálenost:

$$ \frac{2}{n(n-1)} \sum_{i=1}^{n} \sum_{j>i}^{n} d(v_i,v_j) $$

a průměr:

$$ d(G) = \max_{i} \{ e(v_i) \} = \max_{i,j} \{ d(v_i, v_j) \} $$

Budete potřebovat [Floydův algoritmus](https://www.youtube.com/watch?v=4OQeCuLYj-4). Dále určete closeness centralitu
každého vrcholu:

$$ C(i) = \frac{1}{l_i} = \frac{n}{\sum\limits_{j} d_{ij}} $$

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

## Cvičení 8

- Cvičení - Vygenerujte síť dle modelu prefernčního připojování s n = 5000 (počet vrcholů) a m = 2 (počet hran pro každý
  nový vrchol).
    - Vytvořte tři různé vzorky o velikosti 15% původního počtu vrcholů. Tj. použijte tři různé vzorkovací metody.
    - Buď si je naimplementujte sami nebo můžete využít knihovnu
        - [zde](https://github.com/Ashish7129/Graph_Sampling). Pozor jsou tam chyby!
        - nebo [tuto](https://github.com/benedekrozemberczki/littleballoffur).
    - Vizuálně porovnejte distribuci stupňů pro původní síť a pro všechny 3 vzorky (viz snímek 27 z přednášy).
    - Určete D-value podle KS testu - porovnejte distribuci stupňů původní sítě s distribuci stupňů každého vzorku (např
      scipy ks_2samp()).

## Cvičení 9

- Cvičení 1. úkol:

    - Generování dat - vygenerujte pět různých datových sad s těmito rozděleními: normální, exponenciální, mocninné,
    - Poissonovo a lognormální (např. knihovna numpy a např numpy.random.normal()).
    - Fitování rozdělení - např. scipy.stats.norm.fit() pro normální rozdělení a powerlaw.Fit() pro mocninné rozdělení.
    - Vizualizace - histogram každého datasetu s překrytou fitovanou hustotou.
    - Statistická validace - použijte KS-test (scipy.stats.kstest()).

- Cvičení 2. úkol:

    - Vygeneruj dvě sítě, každou s 5500 vrcholy. Jedna bude náhodná podle modelu náhodného grafu a druhá podle modelu
      prefernčního připojování.
    - Vygeneruj a vykresli distribuci stupňů v lineárním měřítku, v logaritmickém měřítku. Pozor na logaritmus nuly.
    - Vvygeneruj a vykresli také CDF a CCDF.
    - Pro distribuce stupňů obou sítí a CCDF proveď fitování Poissonova, normálního, exponenciálního a mocninného
      rozdělení.
    - Vykresli CCDF s překrytými fitovanými rozděleními.

## Cvičení 10

Odolnost

1. Pro síť s desetitisíci vrcholy vygenerovanou podle modelu preferenčního připojování s m = 2
   určete:
    - Počet komponent souvislosti
    - Velikost největší komponenty souvislosti (měřenou počtem vrcholů)
    - Průměrnou vzdálenost (průměrnou délku nejkratší cesty) spočítanou takto:
      $d_L = \frac{\sum_i \sum_{j \gt i} d(v_i, v_j)}{\binom{n}{2}} =
      \frac{2}{n(n-1)} \sum_i \sum_{j \gt i} d(v_i, v_j)$
    - Průměrný stupeň
2. Simulujte odolnost sítě proti rozpadu
    - Cíleným odebíráním vrcholů s největším stupněm
    - Odebíráním náhodných vrcholů. Náhodně odebírejte vrcholy tak dlouho, dokud
      nedosáhnete stejného stavu rozpadu sítě jako odebíráním dle 2a). Nicméně oba
      způsoby odebírání můžete provádět „až do konce“.
3. Po každém odebrání vrcholu podle 2a) nebo 2b) změřte vlastnosti uvedené v bodu 1.
4. Výsledky znázorněte graficky (viz grafy na snímcích 37 a 38) v prezentaci z přednášky) –
   velikost největší komponenty souvislosti, průměrná vzdálenost, průměrný stupeň.
   V jednom obrázku bude vždy jak simulace útoku, tak i simulace selhání pro použitou
   datovou sadu.
