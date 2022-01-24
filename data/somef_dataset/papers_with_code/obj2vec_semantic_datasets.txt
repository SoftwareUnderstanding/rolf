# datasets_similarity

В этом репозитории должны находиться попарные сходства. Всё остальное должно быть в других репозиториях.

Данные, описанные в работе Анны Потапенко(http://www.frccsc.ru/sites/default/files/docs/ds/002-073-05/diss/22-potapenko/ds05_22-potapenko_main.pdf) — стр. 107:

I. WordSim353

- Файлы: wordsim353.zip и wordsim353.ipynb
- Ссылка: оригинал - http://gabrilovich.com/resources/data/wordsim353/ с делением на похожие и связанные - http://alfonseca.org/eng/research/wordsim353.html
- Язык: английский, есть мультиязычные адаптации (немецкий, итальянский, русский): https://leviants.com/multilingual-simlex999-and-wordsim353/
- Количество слов: 153 строки в формате <слово, слово, оценка>, оценка в пределах [0, 10]
- Теги: в репозитории без тегов, есть вариант с тегами http://alfonseca.org/eng/research/wordsim353.html (теги следующие: identical tokens, synonym, antonym, hyponym, hyperonym, sibling terms, first is part of the second one, second is part of the first one, topically related)
- Дополнительно: первоначальным недостатком датасета была нечувствительность к разнице похожих (similarity) и связанных (relatedness) слов. Подробнее в ведении https://arxiv.org/pdf/1408.3456v1.pdf Есть вариант с делением на similar и related: http://alfonseca.org/eng/research/wordsim353.html (см. wordsim_relatedness_goldstandard.txt и wordsim_similarity_goldstandard.txt) 

II. MEN

- Файлы: MEN.zip
- Ссылка: https://staff.fnwi.uva.nl/e.bruni/MEN
- Язык: английский
- Количество слов: 3000 пар слов, строки в формате <слово, слово, оценка>, оценка в пределах [0, 50]
- Теги: есть вариант с тегами части речи (см. MEN_dataset_lemma_form_full)
- Дополнительно: экспертам не объяснялась разница между похожестью и связанностью слов, просили оценить именно связанность, считая похожесть ее частным случаем. Есть предварительное деление на обучающую и тестовую выборки.

III. SimLex-999

- Файлы: simlex999_rus_without_dupl.csv
- Ссылка: https://fh295.github.io/simlex.html
- Язык: английский, есть мультиязычные адаптации (немецкий, итальянский, русский): https://leviants.com/multilingual-simlex999-and-wordsim353/
- Количество слов: 666 Noun-Noun pairs, 222 Verb-Verb pairs and 111 Adjective-Adjective pairs, оценка в пределах [0, 10]
- Теги: часть речи, рейтинги конкретности обеих слов (concreteness rating), сила ассоциации между словами (The strength of free association from word1 to word2), стандартное отклонение всех оценок экспертов на этой паре (The standard deviation of annotator scores when rating this pair) - можно использовать для оценки уверенности в оценке близости.
- Дополнительно: SimLex-999 заточен именно под оценивание похожести слов (similarity), нежели их связанности (relatedness), т.е. решена проблема WordSim353. Например, для similar слов 'coast' - 'shore' оценки	9.00 (SimLex-999)	9.10 (WordSim353), а для related слов 'clothes' - 'closet' оценки	1.96 (SimLex-999)	8.00 (WordSim353).

IV. Mechanical Turk

Файлы: sim-eval-master.zip

This procedure reuses 4 common openly available datasets MC (Miller and Charles, 91), RG (Rubenstein and Goodenougth, 1965), WordSim353 (Finkelstein et al., 2001), BLESS (Baroni and Lenci, 2011)

V. HG-RUS

Файлы: russe-evaluation-master.zip

To Do:
Добавить ссылки на происхождение датасетов

Links:
https://rusvectores.org/static/testsets/

Работы Остроухова и Никитина


Дополнить сведениями, указаны ли части речи и в каких формах указываются слова
Дополнить сведениями о симметричности
