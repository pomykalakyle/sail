// Use reflection to access private[spark] XORShiftRandom
val clazz = Class.forName("org.apache.spark.util.random.XORShiftRandom")
val constructor = clazz.getConstructor(classOf[Long])

for (seed <- Seq(1L, 24L)) {
  val rng = constructor.newInstance(seed.asInstanceOf[java.lang.Long]).asInstanceOf[java.util.Random]
  println(s"seed=$seed:")
  for (i <- 0 until 5) {
    println(s"  $i: ${rng.nextDouble()}")
  }
}
System.exit(0)
