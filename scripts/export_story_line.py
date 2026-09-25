import io, os

BS = chr(92)

INLINE = [
    ('BF:85,BD:99', '[Brglez & Fujiwara, ISCAS 1985; Brglez & Drechsler, ISCAS 1999]'),
    ('S:81',        '[Smith, IEEE Trans. Computers 1981]'),
    ('BK:1999',     '[Brayton & Khatri, VLSI Design 1999]'),
    ('WGT+:2008',   '[Wille et al., RevLib, ISMVL 2008]'),
    ('HS:2000',     '[Hoos & Stuetzle, SATLIB, SAT 2000]'),
    ('R:1991',      '[Reinelt, TSPLIB, ORSA J. Computing 1991]'),
    ('TB:05',       '[Teng & Bolton, CCECE 2005]'),
    ('XX:2024',     '[Abd-El-Barr & Khan, Int. J. Electronics 2014]'),
    ('BBMG:2025',   '[Bos et al., REBEL-6, ISMVL 2025]'),
    ('D:2026',      '[Drechsler, ISMVL 2026]'),
]

REFS = '''References

Verified from the source on 25 September 2026. Each entry below was read from the publisher's
own record, not reconstructed from memory.

[Smith 1981]
   K. C. Smith, "The Prospects for Multivalued Logic: A Technology and Applications View",
   IEEE Transactions on Computers, vol. C-30, no. 9, pp. 619-634, September 1981,
   doi:10.1109/TC.1981.1675860.
   The IEEE record lists the author as "Smith" without initials; the paper is hosted on
   K. C. Smith's own page at the University of Toronto, which settles the initials.

[Wille et al., RevLib, ISMVL 2008]
   R. Wille, D. Grosse, L. Teuber, G. W. Dueck and R. Drechsler, "RevLib: An Online Resource
   for Reversible Functions and Reversible Circuits", 38th International Symposium on
   Multiple Valued Logic (ISMVL 2008), Dallas, TX, USA, 2008, pp. 220-225,
   doi:10.1109/ISMVL.2008.43.

[Hoos & Stuetzle, SATLIB, SAT 2000]
   H. H. Hoos and T. Stuetzle, "SATLIB: An Online Resource for Research on SAT", in SAT 2000,
   I. P. Gent, H. van Maaren and T. Walsh (eds.), IOS Press, 2000, pp. 283-292.

[Reinelt, TSPLIB, ORSA J. Computing 1991]
   G. Reinelt, "TSPLIB -- A Traveling Salesman Problem Library", ORSA Journal on Computing,
   vol. 3, no. 4, pp. 376-384, 1991, doi:10.1287/ijoc.3.4.376.

[Teng & Bolton, CCECE 2005]
   D. H. Y. Teng and R. J. Bolton, "Performance evaluation of multiple-valued logic circuits
   using statistical approach", Canadian Conference on Electrical and Computer Engineering,
   Saskatoon, SK, Canada, 2005, pp. 300-303, doi:10.1109/CCECE.2005.1556932.
   Its abstract opens: "Since there are no standard benchmark functions available for
   comparing multiple-valued logic (MVL) designs, benchmark functions for binary logic design
   are often used for performance analysis of MVL circuits."

[Bos et al., REBEL-6, ISMVL 2025]
   S. Bos, V. Bodahl, O. C. Moholth and H. Gundersen, "REBEL-6: A 32-trit balanced ternary
   instruction set architecture with R2R compiler pipeline for C", 2025 IEEE 55th
   International Symposium on Multiple-Valued Logic (ISMVL), Montreal, QC, Canada, 2025,
   pp. 98-103, doi:10.1109/ISMVL64713.2025.00028.
   The authors are the Ternary Research Group at USN Kongsberg, the host of ISMVL 2027.

[Abd-El-Barr & Khan, Int. J. Electronics 2014]
   M. I. Abd-El-Barr and E. A. Khan, "Improved direct cover heuristic algorithms for synthesis
   of multiple-valued logic functions", International Journal of Electronics, vol. 101, no. 2,
   pp. 271-286, 2014, doi:10.1080/00207217.2013.780296.
   It states: "The first consists of 50,000 2-variable 4-valued randomly generated functions
   and the second consists of 50,000 2-variable 5-valued randomly generated functions", and
   calls them "our benchmarks". No seed is given and the set is not distributed. The paper
   compares against Besslich (1986), Dueck and Miller (1987) and Yang and Wang (1990) by
   running them on its own set, which is sound within the paper; what cannot be done is
   comparing its averages with those published elsewhere on a different 50,000 functions.

[Brglez & Fujiwara, ISCAS 1985]
   F. Brglez and H. Fujiwara, "A neutral netlist of 10 combinational circuits and a target
   translator in Fortran", Int'l Symp. on Circuits and Systems, Special Session on ATPG and
   Fault Simulation, 1985, pp. 663-698.
   Not held by IEEE Xplore or dblp -- ISCAS 1985 predates their coverage -- so no publisher
   record exists to check it against. Google Scholar records the same page range. The entry is
   yours and is the standard form; it is not a conventional paper but a reference to the tape
   distributed at that special session, which is why page ranges differ across citations.

[Brglez & Drechsler, ISCAS 1999]
   F. Brglez and R. Drechsler, "Design of experiments in CAD: context and new data sets for
   ISCAS'99", 1999 IEEE International Symposium on Circuits and Systems (ISCAS), Orlando, FL,
   USA, 1999, pp. 424-427 vol. 6, doi:10.1109/ISCAS.1999.780185.

[Brayton & Khatri, VLSI Design 1999]
   R. K. Brayton and S. P. Khatri, "Multi-valued logic synthesis", Proceedings Twelfth
   International Conference on VLSI Design, Goa, India, 1999, pp. 196-205,
   doi:10.1109/ICVD.1999.745148.
   A survey of multi-valued synthesis, not of decision diagrams; the story line now cites it
   for encoded binary variables, which is what it supports and how you used the key.

Still open

[Drechsler, ISMVL 2026]
   R. Drechsler, "LLM-based Generation of High-Level Benchmarks for MVL Designs", ISMVL 2026.
'''

root = 'D:/DE/MVLBenchmark'
src = io.open(os.path.join(root, 'docs/paper/story.md'), encoding='utf-8').read()
body = src.split('## Notes')[0].rstrip()

out = []
for line in body.splitlines():
    t = line.strip()
    if t.startswith('# Story line'):
        out += ['Story line',
                'The MVL Benchmark Library: An Online Resource for Multi-Valued Benchmarks']
    elif t == '---':
        continue
    elif t.startswith('## '):
        out += ['', t[3:], '']
    elif t:
        out.append(t)
    else:
        out.append('')
txt = '\n'.join(out)

for key, readable in INLINE:
    txt = txt.replace(BS + 'cite{' + key + '}', readable)
assert BS + 'cite' not in txt, 'unreplaced key'

txt = txt.rstrip() + '\n\n' + REFS
while '\n\n\n' in txt:
    txt = txt.replace('\n\n\n', '\n\n')

io.open(os.path.join(root, 'output/story-line.txt'), 'w',
        encoding='utf-8', newline='\r\n').write(txt.strip() + '\n')
print('written:', len(txt.splitlines()), 'lines')
