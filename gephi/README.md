# Návod pro práci s Gephi

Gephi je nástroj pro vizualizaci a analýzu sítí. Jeho hlavní výhodou je možnost vizualizace velkých sítí a možnost
interaktivní práce s nimi. Gephi je dostupný pro všechny hlavní operační systémy a je zdarma.
Gephi je možné stáhnout z oficiálních stránek [gephi.org](https://gephi.org/).

## Načtení gephi souboru

File -> Open -> vybrat soubor

## Cvičení

### Cvičení 5

Cílem cvičení bylo vytvořit vizualizaci Iris datasetu na základě původních tříd a nově vytvořených
komunit na základě komunitní modularity, která byla vypočítána pomocí algoritmu Louvain.

Soubor obsahuje tři sítě:

- Síť s hranami vybranými kNN algoritmem s parametrem k=3 a sigma=1.0.
- Síť s hranami vybranými e-radius algoritmem s parametrem epsilon=0.9.
- Síť s kombinací obou předchozích sítí.

Přepínání mezi sítěmi je možné pomocí záložek na horní liště uprostřed. Obarvení uzlů je podle tříd a komunit je možné
přepínat v sekci `Appearance` v horním panelu. Zde vyberte ikonu palety, sekci `Nodes` a v `Partition` vyberte
`Modularity Class` nebo `species`. Potom stačí kliknout na tlačítko `Apply`.