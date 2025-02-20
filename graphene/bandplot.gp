set terminal postscript eps enhanced color font "Helvetica,14"
set output 'bandplot.eps'
set pm3d
set xrange [-4:4]
set yrange [-4:4]
set zrange [-3:3]  # Adjust based on your band structure range
set title "Graphene Band Structure"
set xlabel "kx"
set ylabel "ky"
set dgrid3d 200,200
set hidden3d

splot "bandplot.dat" using 1:2:3 with pm3d title "Valence Band", \
      "bandplot.dat" using 1:2:4 with pm3d title "Conduction Band"


