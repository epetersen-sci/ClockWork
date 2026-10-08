# Example data

Six DAM monitors (`Monitor17.txt` to `Monitor22.txt`, 32 flies each, 192 in
all) and the `metadata.xlsx` that describes them. The flies were recorded for
3 days in LD and then 9 days in DD at 25 °C. The files keep about a week of
extra recording before the window the metadata asks for, and everything after
it, so that ClockWork's trimming has something to do.

## Source

These recordings come from the work published in:

> Richhariya, S., Shin, D., Schlichting, M. & Rosbash, M. (2025) Metabolic
> rewiring prevents neurodegeneration caused by chronic mitochondrial
> dysfunction. *Current Biology* 35(22):5443–5459.e5.
> <https://doi.org/10.1016/j.cub.2025.09.063>

They are included with Shlesha Richhariya's permission, as a real experiment to
learn ClockWork on. If you use them in anything other than learning or testing
ClockWork, cite the paper above.

## Genotypes

One genotype per monitor:

| Monitor | `genotype` |
|---|---|
| 17 | `dsmcherry` |
| 18 | `dsmcherry+Ldhmut` |
| 19 | `dsOpa1(67159)` |
| 20 | `dsOpa1(67159)+Ldhmut` |
| 21 | `dsOpa1(32358)` |
| 22 | `dsOpa1(32358)+Ldhmut` |

`dsmcherry` is the mCherry control and `dsOpa1(…)` are two independent *Opa1*
lines, identified by stock number. Each line is shown alone and together with
the *Ldh* mutation (`+Ldhmut`). The paper gives the full genotypes and how the
flies were made.
