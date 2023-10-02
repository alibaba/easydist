export ENABLE_GRAPH_COARSEN="True"
export COARSEN_LEVEL=0

torchrun --nproc_per_node 2 --master_port 36543 simple_model1.py |& tee 1n2g.log

expected_cost=330.0
costs=`grep -e "\[Communication Cost\]:[^\n]*" 1n2g.log -o | grep -e "[0-9.]*" -o | awk '{print $1}'`;
cost=`echo $costs | awk '{print $1}'`;
echo -e "\n*****************************************"
echo "Communication cost is $cost"

if [ `echo "${cost} > ${expected_cost}"|bc` -eq 1 ];
then echo -e "Failed!\nExpected communication cost is ${expected_cost}."
else echo "Successful!"
fi
echo -e "*****************************************\n"

