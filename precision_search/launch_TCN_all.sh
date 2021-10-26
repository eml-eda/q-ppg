#*----------------------------------------------------------------------------*
#* Copyright (C) 2021 Politecnico di Torino, Italy                            *
#* SPDX-License-Identifier: Apache-2.0                                        *
#*                                                                            *
#* Licensed under the Apache License, Version 2.0 (the "License");            *
#* you may not use this file except in compliance with the License.           *
#* You may obtain a copy of the License at                                    *
#*                                                                            *
#* http://www.apache.org/licenses/LICENSE-2.0                                 *
#*                                                                            *
#* Unless required by applicable law or agreed to in writing, software        *
#* distributed under the License is distributed on an "AS IS" BASIS,          *
#* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   *
#* See the License for the specific language governing permissions and        *
#* limitations under the License.                                             *
#*                                                                            *
#* Author:  Alessio Burrello                                                  *
#*----------------------------------------------------------------------------*

folder="test_28_07"
echo $folder

mkdir $folder

echo

networks="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 "

if [ $1 == 'MN-dilht' ] && [ $2 == '8' ]
then
for net in $networks;
do
	echo $1 $2
	echo $net
	python3 test_TEMPONetDaliaTrainer.py --gpu 1 --cross-validation True --finetuning True --sheet MN-dilht --net_number $net --quantization 8 > ./$folder/cr_val_quant_8_MN_dilht_$net.txt
done
fi

if [ $1 == 'MN-dilht' ] && [ $2 == '4' ]
then
for net in $networks;
do
	echo $1 $2
	echo $net
	python3 test_TEMPONetDaliaTrainer.py --gpu 2 --cross-validation True --finetuning True --sheet MN-dilht --net_number $net --quantization 4 > ./$folder/cr_val_quant_4_MN_dilht_$net.txt
done
fi

if [ $1 == 'MN-dilht' ] && [ $2 == '2' ]
then
for net in $networks;
do
	echo $1 $2
	echo $net
	python3 test_TEMPONetDaliaTrainer.py --gpu 3 --cross-validation True --finetuning True --sheet MN-dilht --net_number $net --quantization 2 > ./$folder/cr_val_quant_2_MN_dilht_$net.txt
done
fi


networks="1 2 3 4 5 6 7 8 9 10 11 "

if [ $1 == 'PIT' ] && [ $2 == '8' ]
then
for net in $networks;
do
	echo $1 $2
	echo $net
	python3 test_TEMPONetDaliaTrainer.py --gpu 1 --cross-validation True --finetuning True --sheet PIT --net_number $net --quantization 8 > ./$folder/cr_val_quant_8_PIT_$net.txt
done
fi

if [ $1 == 'PIT' ] && [ $2 == '4' ]
then
for net in $networks;
do
	echo $1 $2
	echo $net
	python3 test_TEMPONetDaliaTrainer.py --gpu 2 --cross-validation True --finetuning True --sheet PIT --net_number $net --quantization 4 > ./$folder/cr_val_quant_4_PIT_$net.txt
done
fi

if [ $1 == 'PIT' ] && [ $2 == '2' ]
then
for net in $networks;
do
	echo $1 $2
	echo $net
	python3 test_TEMPONetDaliaTrainer.py --gpu 3 --cross-validation True --finetuning True --sheet PIT --net_number $net --quantization 2 > ./$folder/cr_val_quant_2_PIT_$net.txt
done
fi