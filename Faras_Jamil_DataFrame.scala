import org.apache.spark.SparkConf
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.functions.desc

import scala.util.control.Breaks._

object Fjamil {

  /**
    * Constants
    */

  val ATTRIBUTES: Array[String] = Array("verifiedPurchase",
    "vine",
    "reviewBody",
    "starRating")
  val TARGET: String = "helpfulReview"
  val THRESHOLD = 10000

  /**
    * Main method
    */

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf()
    conf.setAppName("Faras Jamil test1")

    conf.setMaster("local[4]") //remove this line when running on cluster

    val ss = SparkSession.builder().config(conf).getOrCreate()
    import ss.implicits._

    ss.sparkContext.setLogLevel("WARN")

    val t1 = System.nanoTime

    val rdd = ss.sparkContext.textFile("/Users/farisjameel/Documents/MS/semester3/CCBD/BigData/project/data/") //sample_us.tsv
    //    val rdd = ss.sparkContext.textFile("hdfs://10.0.0.1:54310/data/amazon/")
    val header = rdd.first()
    val data = rdd.filter(_ != header)
    val df = data.map(parseReview).toDF.persist()
    println("df",df.rdd.getNumPartitions)
    val splittedDf = df.randomSplit(Array(0.7, 0.3))
    val trainDf = splittedDf(0)
    val testDf = splittedDf(1)
    println("traindf",trainDf.rdd.getNumPartitions)

    val tree = ID3(trainDf, ATTRIBUTES)

    val prediction = testDf.map(row => (predict(tree, parseReviewFromRow(row)))).persist()
    val correctPredictions = prediction.where(prediction("value") === true).count()
    val totalRecords = prediction.count()
    val accuracy = (correctPredictions.toFloat / totalRecords.toFloat) * 100
    println("Accuracy " + accuracy + "%")


    val duration = (System.nanoTime - t1) / 1e9d
    println("Elapsed time: " + duration)

    System.in.read()

  }

  /**
    * Review parsing
    */

  def possibleValues(attribute: String): Array[Any] = {
    attribute match {
      case "starRating" => return Array(1, 2, 5)
      case "reviewBody" => return Array(1, 2, 3)
      case _ => return Array(true, false)
    }
  }

  def getNodeValue(attribute: String, review: Review): Any = {
    attribute match {
      case "starRating" =>
        return review.starRating
      case "vine" =>
        return review.vine
      case "verifiedPurchase" =>
        return review.verifiedPurchase
      case "reviewBody" =>
        return review.reviewBody
      case "helpfulReview" =>
        return review.helpfulReview
    }
  }

  case class Review(
                     starRating: Int,
                     helpfulReview: Boolean,
                     vine: Boolean,
                     verifiedPurchase: Boolean,
                     reviewBody: Int
                   )

  def stringToBoolean(yesNo: String) = if (yesNo.equalsIgnoreCase("y")) true else false

  def parseReview(row: String): Review = {
    val review = row.split("\t")
    return Review(
      parseStarRating(review(7)), // star_rating
      parseHelpfulReview(review(8), review(9)), // helpful_votes, total_votes
      stringToBoolean(review(10)), // vine
      stringToBoolean(review(11)), // verified_purchase
      parseReviewBody(review(13)) // review_body
    )
  }

  def parseReviewFromRow(review: Row): Review = {
    Review(
      review.get(0).toString.toInt,
      review.get(1).toString.toBoolean,
      review.get(2).toString.toBoolean,
      review.get(3).toString.toBoolean,
      review.get(4).toString.toInt
    )
  }

  def parseStarRating(ratingString: String): Int = {
    // Exception handling in case ratingString don't have a number
    var rating = 5
    try {
      rating = ratingString.toInt
    } catch {
      case e: NumberFormatException => println("rating number not found" + e)
    }
    //    rating

    if (rating == 5)
      return 5
    else if (rating == 1)
      return 1
    return 2
  }

  def parseHelpfulReview(helpfulVotesString: String, totalVotesString: String): Boolean = {
    // Exception handling in case helpfulVotesString and totalVotesString don't have a number
    var helpfulVotes, totalVotes = 0
    try {
      helpfulVotes = helpfulVotesString.toInt
      totalVotes = totalVotesString.toInt
    } catch {
      case e: NumberFormatException => println("rating number not found" + e)
    }
    if (totalVotes == 0) return false
    if ((helpfulVotes.toFloat / totalVotes.toFloat) > 0.5) true else false
  }

  def parseReviewBody(reviewBody: String): Int = {
    val textSize = reviewBody.length

    if (textSize < 100)
      return 1
    else if (textSize >= 100 && textSize < 300)
      return 2
    else
      return 3
  }

  /**
    * Decision Tree Data Structure
    */

  trait Tree

  case class Leaf(value: Any) extends Tree

  case class Node(attribute: String) extends Tree {
    var nodes: Array[NodeData] = Array()

    def addSubTree(value: Any, node: Tree): Unit = {
      this.nodes :+= NodeData(node, value)
    }
  }

  case class NodeData(tree: Tree, value: Any)

  /**
    * Decision Tree implementation
    */

  def ID3(data: DataFrame, attributes: Array[String]): Tree = {
    data.persist()
    val count = data.count()
    var entropy: Double = 0
    if (count > THRESHOLD)
      entropy = H(data, TARGET, count)

    if (count < THRESHOLD || attributes.isEmpty || entropy == 0.0) {
      if (count == 0) return null
      val leaf = Leaf(mostCommonValue(data, TARGET))
      return leaf
    } else {
      val ig = attributes.map(attr => (attr, IG(data, TARGET, attr, entropy, count)))
      val maxIGAttribute = ig.maxBy(_._2)._1

      // remove selected attribute from attributes list
      val new_attributes = attributes.filter(_ != maxIGAttribute)

      println("create new tree with " + maxIGAttribute)

      val tree = new Node(maxIGAttribute)
      (maxIGAttribute, possibleValues(maxIGAttribute))

      for (value <- possibleValues(maxIGAttribute)) {
        val filteredData = data.filter(data(maxIGAttribute) === value).drop(maxIGAttribute)
        val subTree = ID3(filteredData, new_attributes)
        tree.addSubTree(value, subTree)
      }
      return tree
    }
  }

  def H(data: DataFrame, target: String, count: Long): Double = {
    val values = possibleValues(target)
    var prob: Array[Double] = Array()
    for (value <- values) {
      val p: Double = data.select(target).where(data(target) === value).count().toDouble / count
      if (p == 0)
        prob :+= 0.0
      else
        prob :+= -p * (math.log(p) / math.log(2))
    }
    return prob.sum
  }

  def IG(data: DataFrame, target: String, attribute: String, classEntropy: Double, count: Long): Double = {
    val values = possibleValues(attribute)
    val targets = possibleValues(target)
    var entropy: Array[Double] = Array()
    for (value <- values) {
      var prob: Array[Double] = Array()
      val c: Double = data.select(attribute).where(data(attribute) === value).count().toDouble
      breakable {
        for (t <- targets) {
          if (c == 0)
            break
          val p: Double = data.select(attribute, target).where(data(target) === t && data(attribute) === value).count().toDouble / c
          if (p == 0)
            prob :+= 0.0
          else
            prob :+= -p * (math.log(p) / math.log(2))
        }
      }
      entropy :+= prob.sum * c / count
    }

    return (classEntropy - entropy.sum)
  }

  def mostCommonValue(data: DataFrame, target: String): Any = {
    return data.groupBy(target).count.orderBy(desc("count")).first().get(0)
//    val values = possibleValues(target)
//    var counts: Array[(Any, Long)] = Array()
//    for (value <- values) {
//      counts :+= (value, data.select(target).where(data(target) === value).count())
//    }
//
//    return counts.maxBy(_._2)._1

  }

  /**
    * Prediction
    */

  def predict(tree: Tree, review: Review): Boolean = {
    var t = tree
    while (t.isInstanceOf[Node]) {
      val node: Node = t.asInstanceOf[Node]
      val v = getNodeValue(node.attribute, review)
      t = node.nodes.filter(_.value == v)(0).tree
    }
    if (t.isInstanceOf[Leaf]) {
      val leaf: Leaf = t.asInstanceOf[Leaf]
      if (leaf.value == null) return false
      return if (leaf.value.toString.toBoolean == getNodeValue(TARGET, review)) true else false
    }
    return false
  }
}
