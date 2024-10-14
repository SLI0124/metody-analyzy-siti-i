# Metody analýzy sítí

Cílem předmětu je představení komplexních sítí se zaměřením na jejich typy (sociální, komunikační, biologické apod.),
vlastnosti, modely a na metody jejich analýzy. Po absolvování předmětu bude student rozumět principům, které ovlivňují
vlastnosti sítí, bude schopen aplikovat metody související s analýzou těchto vlastností a bude schopen prototypově
implementovat vybrané metody a modely. Dále bude schopen využít nástroje a knihovny pro analýzu a vizualizaci sítí
a po aplikaci metod analýzy sítí bude umět posoudit relevanci výsledků a nalézt jejich srozumitelnou interpretaci.

# Cvičení

## Cvičení 1

První analýza síťových dat - implementujte vše sami, stačí konzolová aplikace. Data z kolekce Karate
Club reprezentujte maticí sousednosti (adjacency matrix) a seznamem vrcholů a jejich sousedů (adjacency list). Určete
min, max a průměrný stupeň vrcholů. Určete četnost a relativní četnost výskytů vrcholů s daným stupněm a vytvořte
histogram těchto četností (sloupcový graf, kde na ose x je stupeň a na ose y počet vrcholů s daným stupněm).

## Cvičení 2

Zadání cvičení - implementujte vše sami, stačí konzolová aplikace. Pro Karate Club určete vzdálenost (délku nejkratší
cesty) mezi všemi dvojicemi vrcholů, průměrnou vzdálenost a průměr. Budete potřebovat Floydův algoritmus. Dále určete
closeness centralitu každého vrcholu

$$C(i) = \frac{n}{\sum_{j=1}^{n} d(i,j)}$$

## Cvíčení 3

Pro Karate Club určete shlukovací koeficient každého vrcholu a určete tranztivitu sítě (tedy průměrný shlukovací
koeficient). Určete shlukovací efekt. Ten se určí jako průměrný CC pro vrcholy daného stupně. Distribuci tohoto
průměrného CC (osa Y) vůči stupni (osa X) vykreslete (výsledek). Výsledky měření všech lokálních vlastností uložte do
CSV souboru. Ten bude obsahovat následujcí sloupce: ID vrcholu, jeho stupeň, closeness centralitu a shlukovací
koeficient.

## Datasety

- [Zachary's karate club](https://websites.umich.edu/~mejn/netdata/karate.zip)