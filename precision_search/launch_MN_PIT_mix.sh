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

folder="test_mixed"
echo $folder

mkdir $folder
if [ $3 == '0.0001' ]
then
python3 test_TEMPONetDaliaTrainer.py --gpu $4 --cross-validation False --finetuning False --sheet $1 --cd $3 --net_number $2 --quantization mix-search > ./$folder/cr_val_$1\_$2\_cd_med.txt
fi

if [ $3 == '0.00001' ]
then
python3 test_TEMPONetDaliaTrainer.py --gpu $4 --cross-validation False --finetuning False --sheet $1 --cd $3 --net_number $2 --quantization mix-search > ./$folder/cr_val_$1\_$2\_cd_small.txt
fi

if [ $3 == '0.001' ]
then
python3 test_TEMPONetDaliaTrainer.py --gpu $4 --cross-validation False --finetuning False --sheet $1 --cd $3 --net_number $2 --quantization mix-search > ./$folder/cr_val_$1\_$2\_cd_big.txt
fi