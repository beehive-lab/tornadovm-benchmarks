tornado --jvm="-Dtornado.device.memory=2GB -Dbenchmark.mxm.device=0:2" -cp target/tornadovm-benchmarks-1.0-SNAPSHOT.jar tornadovm.benchmarks.Main $@
