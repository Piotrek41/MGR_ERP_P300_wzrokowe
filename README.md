# MGR_ERP_P300_wzrokowe --- Opis i działanie skryptów

## Wymagane biblioteki
Wymagane biblioteki są zawarte w pliku *\requirements.txt*
$ pip install -r requirements.txt command 

## Streszczenie

\- ***run_mp_on_epofif.py*** --- sposób wywołania algorytmu MMP1 na
plikach *\*.fif*; algorytm MMP1 wczytywany jest z pliku wykonawczego
***empi-lin64***

\- ***read_MP_books_and_search_for_iter_with_stat.py ---*** wczytanie i
odpowiednia segregacja atomów z bazy danych *\*.db* utworzonej przez
skrypt ***run_mp_on_epofif.py***

\- ***fitting_dipol_to_mp_books.py* ---** wczytanie, odpowiednia
segregacja atomów i głównie dopasowanie dipoli do atomów z bazy danych
*\*.db* utworzonej przez skrypt ***run_mp_on_epofif.py***

\- ***reading_and_showing_dipol.py** ---* wczytanie, roszerzenie (o
odległość dipola od kory) i wizualizacja baz danych *\*.pkl* z dipolami
utworzonych przez skrypt ***fitting_dipol_to_mp_books.py***

## Funkcje i pomocnicze skrypty

\- ***empi-lin64*** --- skompilowana pod linux wersja alogorytmu empi
v1.0 w formie pliku wykonywawczego ---
<https://github.com/develancer/empi>

\- niektóre funkcje zawarte w plikach oraz skrypty:
***cortical_distance.py, helper_functions.py,
interpolate_head_features.py, miscellaneous.py, mne_conversions.py,
mne_freesurf_labels.py*** i foldery ***brain, obci, kernels*** zostały
zaciągnięte z paczek: <https://gitlab.com/BudzikFUW/budzik-analiza-py-3>
, <https://gitlab.com/BudzikFUW/coma_structures> . Znajdzujące się w tym projekcie pliki i
zaciągniete funkcje mogły zostać poddane modyfikacjom.


## Szczegóły
### 1. run_mp_on_epofif.py

Skrypt przetwarza dane podane w postaci listy ścieżek do plików
*\*.fif*. Dane z tej listy są ze sobą łączone --- lista może być jedno
elementowa. Następnie skrypt jednocześnie przepróbowuje dane do cztery
razy mniejszej częstości i przekształca je wedle wymagań algorytmu MMP1.
Między innymi algorytmowi podawany jest jednowymiarowy wektor zwierający
najpierw odcinki target zachowując przy tym kolejność kanałów i odcinki
nontarget również w kolejności kanałów. Zachowanie odpowiedniej
chronologii odcinków ułatwia później obchodzenie się z danymi.

Tak podane dane po przejściu przez algorytm zapisywane są w postaći bazy
danych w formacie *\*.db*, w nowo utworzonym folderze **mp_books** w
podfolderze ścieżki z obecenie przetwarzanymi plikami *\*.fif*.
Utworzony w folderze ***mp_books*** również zostanie plik textowy
*\*.txt*, który będzie wykorzystywany między innymi w skryptach*:
**fitting_dipol_to_mp_books.py,
searching_for_relevant_iterations_with_tests.py**.* Ten plik tekstowy
zawiera informacje o:

-parametrach podanych algorytmowi mp

\- "-f" --- częstość próbkowania

\- "-c" --- łączna liczba odcinków (target i nontarget)

-liczbę i nazwy rozkładanych elektrod

-długość przetwarzanych odcinków

-liczbę odcinków target i nontarget na poszczególnej elektrodzie

Skrypty **s*earching_for_relevant_iterations_with_tests.py,
fitting_dipol_to_mp_books.py ***rozszerzają na pewnym etapie wczytaną
bazę danych o nazwy poszczególnych elektrod i numery odcinków na
podstawie pliku tekstowego oraz dzieli ją na odcinki target i nontarget.
Tak rozszerzona baza danych nie jest nadpisywana we wczytywanym pliku.

**Uwagi**

Skrypt wykonuje się na liście ze ścieżkami do plików *\*.fif* , jest on
napisany w taki sposób, aby łączył i wczytywał pliki zgodnie z bazą
danych ***Measurements_database_merged.csv.*** Ścieżka do pliku *\*.csv*
została podana bespośrednio w kodzie, należy wczytywanie tego pliku
dostosować indywidualnie np. z wykorzytaniem moduł *sys.argv* i podaniem
ścieżek z linii poleceń podczas wywołania skryptu.

Podczas wywołania skryptu w lini poleceń jako pierwszy argument po
nazwie programu należy podać ścieżkę do folderu z danymi *\*.fif* ---
może zostać podana ścieżka do folderu zawierającego kilka pod-folderów z
danymi.

**Przykład wywołania skryptu**

$ python3 run_mp_on_epofif.py /mnt/c/Users/Piotr/Desktop/p300-wzrokowe

### 2. read_MP_books_and_search_for_iter_with_stat.py

Skrypt wczytuje pliki \*.db utworzone skryptem **run_mp_on_epofif.py.**
Odpowiednio przetwarza dane wykorzytując wspomniany w punkcie "1" plik
tekstowy, w celu rozrózniania odcinków w zależności od elektrody, numeru
odcinka, przynależności do target/nontarget.

W skrypcie wybierane są odpowiednie iteracje mające wyodrębniąć
najbardziej różnicujące struktury między target i nontarget na bazie
statystyki w funkcji *statistic()*, mające z założenia oddawać charakter
potencjału wywowałengo P300. Liczony jest również test permutacyjny na
bazie funkcji *statistic()* na sumie energii wybranych struktur, z
którego wartości p są korygowane poprawką Holma-Bonferroniego
wykorzystując moduł *multipletests* z biblioteki *statsmodels*, aby
potencjalnie rozwiązać problem wielokrotnych porównań. Na sumach energii
liczona jest równieża sama wartość statystyki z funkcji *statistic().*
Wyniki odpowiednich obliczeń są zapisywane do pilków tekstowych.

**Uwagi**

W kodzie zakomentowany jest przykład wykonania testu klastrowego na
wybranych strukturach, przykład wizualizacji średniej sumy wybranych
struktur na tle rekonstrukcji ERP, czyli średniej sumy ze wszystkich
struktur. Zmienna *outdir* w kodzie określa ścieżkę do folderu, który
zostanie utworzony i do niego zostaną zapisane wyżej wspomniane obrazki
*\*.png* oraz piliki tesktowe *\*.txt* . Należy dostosować tą ścieżkę do
siebie lub użyć modułu sys.argv.

Podczas wywołania skryptu w lini poleceń jako pierwszy argument po
nazwie programu należy podać ścieżkę do folderu z danymi *\*.db* ---
może zostać podana ścieżka do folderu zawierającego kilka pod-folderów z
danymi.

**Przykład wywołania skryptu**

$ python3 read_MP_books_and_search_for_iter_with_stat.py /mnt/c/Users/Piotr/Desktop/p300-wzrokowe

### 3. fitting_dipol_to_mp_books.py

Skrypt wczytuje pliki \*.db utworzone skryptem **run_mp_on_epofif.py.**
Odpowiednio przetwarza dane wykorzytując wspomniany w punkcie "1" plik
tekstowy, w celu rozrózniania odcinków w zależności od elektrody, numeru
odcinka, przynależności do target/nontarget. Następnie przypisuje znaki
"-" amplitudom atomów w takiej bazie danych oparte na fazie danego
atomu. Algorytm następnie dopasowuje dipol do każdego z atomów na
podstawie tak pozmienianych amplitud. Policzone parametry dipoli
rozszerzone o parametry atomów do których zostały dopasowane zapisywane
są do plików *\*.pkl*.

Przykład wczytywania i obchodzenia się z danymi z tak utworzonego pliku
*\*.pkl* zawarty jest w skrypcie ***reading_and_showing_dipol.py**.*

**Uwagi**

Skrypt wczytuje diagnozy badanych z pliku
***Measurements_database_merged.csv***, ścieżke do tego pliku należy
zmienić w kodzie lub zastsować moduł sys.argv i podać ją w linii
poleceń.

Podczas wywołania skryptu w lini poleceń jako pierwszy argument po
nazwie programu należy podać ścieżkę do folderu z danymi *\*.db* ---
może zostać podana ścieżka do folderu zawierającego kilka pod-folderów z
danymi. Jako kolejny argument należy podać do których odcinków chcemy
dopasować dipol target czy nontarget.

**Przykład wywołania skryptu**

$ python3 fitting_dipol_to_mp_books.py /mnt/c/Users/Piotr/Desktop/p300-wzrokowe target

### 4. reading_and_showing_dipol.py

Skrypt odpowiednio przetwarza dane wykorzytując plik *\*.pkl* uwtorzony
przez skrypt ***fitting_dipol_to_mp_books.py*** (punkt 2). Tworzy w
folderze z plikiem *\*.pkl* plik *\*.png* z wizualizacją położeń
wczytanych atomów dipolowych dla wybranych iteracji. Tworzony i
zapisywany zostaje również histogram z liczbą atomów dipolowych w
wybranych iteracjach.

Dodatkowo skrypt rozszerza wczytaną bazę danych o odległość dipola wobec
najbliżeszego voksela kory oraz o położenie i etykietę tego voksela.
Plik *\*.pkl* zostaje nadpisany rozszerzoną bazą danych. Ponowne
wywołanie tego skryptu nie spowoduje dodatkowych obliczeń.

**Uwagi**

W funckji main jako trzeci argument należy podać listę z numerami
iteracji do wizualizacji. Algorytm ustawiony jest na wyrysowanie
wszystkich możliwych 15 iteracji z ówcześnie analizowanych danych.
Zakomentowany jest również przykład wczytywania pożądanych list z
numerami iteracji na podstawie pliku tekstowego. Wyrysowywane są atomy
dipolowe o jakości dopasowania GOF = 60 (zmienna gof = 60).

Podczas wywołania skryptu w lini poleceń jako pierwszy argument po
nazwie programu należy podać ścieżkę do folderu z danymi *\*.pkl* ---
może zostać podana ścieżka do folderu zawierającego kilka pod-folderów z
danymi. Jako kolejny argument należy podać czy chcemy wyrysować pliki z
danymi target, czy nontarget.

**Przykład wywołania skryptu**

$ python3 reading_and_showing_dipol.py /mnt/c/Users/Piotr/Desktop/dipols target
