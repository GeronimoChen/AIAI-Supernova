for i in $(seq $2 $3); do
	tardis Yaml/$1/$i.yml Spec/$1/$i.txt
done;
