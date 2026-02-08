

## Suppose you have many cores, and you want a better htop because your terminal space is not big enough.

![one_line_code](one_line_code.png)

This is the julia code that does it (with UnicodePlots.jl). Just run this in bash.

```bash


julia -e 'using UnicodePlots; n=Sys.CPU_THREADS; cols=Int(floor(sqrt(n))); rows=Int(ceil(n/cols)); hist=zeros(n,20); ram_h=Float64[]; p_i=zeros(n); p_t=zeros(n); while true; l=readlines("/proc/stat")[2:n+1]; c_i=[parse(Int,split(ln)[5]) for ln in l]; c_t=[sum(parse.(Int,split(ln)[2:8])) for ln in l]; d_i=c_i.-p_i; d_t=c_t.-p_t; usage=[dt>0 ? 100*(1-di/dt) : 0.0 for (di,dt) in zip(d_i, d_t)]; global p_i=c_i; global p_t=c_t; global hist=hcat(hist[:,2:end], usage); m=readlines("/proc/meminfo"); total=parse(Int,split(m[1])[2]); avail=parse(Int,split(m[3])[2]); push!(ram_h, 100*(1-avail/total)); if length(ram_h)>20 deleteat!(ram_h,1) end; print("\033[H\033[2J"); display(heatmap(transpose(reshape([usage; zeros(rows*cols-n)], cols, rows)), title="Core Heatmap", colormap=:inferno)); plt=lineplot(hist[1,:], title="300-Thread Overlay", ylim=[0,100], width=cols*4, height=7, canvas=DotCanvas); for i in 2:n lineplot!(plt, hist[i,:]) end; display(plt); length(ram_h)>1 && display(lineplot(ram_h, title="RAM Usage %", ylim=[0,100], width=cols*4, height=4, color=:blue)); sleep(1); end'

```



a simpler way to monitor this is to use mpstat (and use awk to organize it to columns)

```
watch -n 1 "mpstat -P ALL 1 1 | awk 'NR>4 {printf \"CPU %s: %s%%\\n\", \$3, \$12}' | pr -10 -t -w $(tput cols)"

```