# coding: utf-8

import time
import os

def wait_for_a_process_to_end(processes):
    """Zabezpiecza przed uruchomieniem zbyt wielu prcosów na raz i przeciążeniem w ten sposób maszyny.
    Funkcja ta pracuje w pętli do czasu, aż nie zakończy się przynajmniej jeden z procesów.

    :param processes: lista procesów
    """
    i = 0
    while True:
        proc = processes[i]
        time.sleep(1)  # sprawdzaj co sekundę kolejne procesy w pętli
        if proc.poll() is not None:  # jeśli dany proces się zakończył
            proc.wait()
            processes.remove(proc)  # to usuń go z listy i kontynuuj główną pętlę
            break
        i = (i + 1) % len(processes)


def n_parent_dirpath(path, n):
    """Funckja zwracająca n-ty w hierarchii macierzysty katalog, w którym leży plik.
    :param path: ścieżka do pliku/katalogu
    :param n: o ile stopni w hierarchii w górę podać katalog (1 to katalog, w którym leży plik, 2 to katalog, w którym leży katalog, w którym leży plik, itd.)
    :return: scieżka do n-tego w hierachii macierzystego katalogu
    """
    for i in range(n):
        path = os.path.dirname(path)
    return path