tornado --jvm="-Dtornado.device.memory=4GB -Dbenchmark.mxm.device=0:0" -cp target/tornadovm-benchmarks-1.0-SNAPSHOT.jar tornadovm.benchmarks.Main $@
